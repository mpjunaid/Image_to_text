import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import string
import pandas as pd

# -------- Config --------
CHARSET = string.ascii_letters + string.digits
CHAR2IDX = {c: i + 1 for i, c in enumerate(CHARSET)}  # 0 is reserved for CTC blank
IDX2CHAR = {i + 1: c for i, c in enumerate(CHARSET)}
BLANK_IDX = 0
IMAGE_HEIGHT = 32
MODEL_PATH = "real_ocr_data/crnn_model_final.pth"

# -------- Utils --------
def encode_text(text):
    return [CHAR2IDX[c] for c in text if c in CHAR2IDX]

def decode_output(pred):
    pred = pred.argmax(2)
    output = []
    for i in range(pred.shape[0]):
        seq = pred[i].detach().cpu().tolist()
        prev = -1
        chars = []
        for idx in seq:
            if idx != prev and idx != BLANK_IDX:
                chars.append(IDX2CHAR.get(idx, ''))
            prev = idx
        output.append("".join(chars))
    return output

# -------- Dataset --------
class RealOCRDataset(Dataset):
    def __init__(self, image_folder, label_file):
        self.image_folder = image_folder
        self.samples = []
        with open(label_file, 'r') as f:
            for line in f:
                name, label = line.strip().split(maxsplit=1)
                self.samples.append((name, label))

        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((IMAGE_HEIGHT, 100)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, text = self.samples[idx]
        image_path = os.path.join(self.image_folder, filename)
        img = Image.open(image_path).convert('RGB')
        img = self.transform(img)
        label = torch.tensor(encode_text(text), dtype=torch.long)
        return img, label

def collate_fn(batch):
    images, labels = zip(*batch)
    image_tensor = torch.stack(images)
    label_lengths = torch.tensor([len(l) for l in labels])
    labels = torch.cat(labels)
    input_lengths = torch.full((len(images),), image_tensor.size(3) // 4, dtype=torch.long)
    return image_tensor, labels, input_lengths, label_lengths

# -------- CRNN Model --------
class CRNN(nn.Module):
    def __init__(self, num_classes):
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.rnn = nn.LSTM(128 * 8, 256, bidirectional=True, num_layers=1, batch_first=True)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        b, c, h, w = x.size()
        x = x.permute(0, 3, 1, 2).reshape(b, w, c * h)
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x.permute(1, 0, 2)  # [T, B, C]

# -------- Evaluation --------
def evaluate_precision(model, dataset, device, name="Dataset"):
    loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels, input_lengths, label_lengths in loader:
            images = images.to(device)
            output = model(images).log_softmax(2)
            pred_texts = decode_output(output)

            start = 0
            true_texts = []
            for length in label_lengths:
                segment = labels[start:start+length]
                true_texts.append("".join([IDX2CHAR[i.item()] for i in segment]))
                start += length

            for pred, true in zip(pred_texts, true_texts):
                if pred == true:
                    correct += 1
                total += 1

    precision = correct / total if total > 0 else 0
    print(f"🔍 Precision on {name}: {precision:.4f} ({correct}/{total} correct)")

# -------- Main --------
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load training data
    train_dataset = RealOCRDataset('real_ocr_data/train/images', 'real_ocr_data/train/labels.txt')
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    # Initialize model
    model = CRNN(num_classes=len(CHARSET) + 1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CTCLoss(blank=BLANK_IDX)

    # Training loop
    print("🚀 Starting training...")
    for epoch in range(100):
        model.train()
        total_loss = 0
        for images, labels, input_lengths, label_lengths in train_loader:
            images, labels = images.to(device), labels.to(device)

            logits = model(images)
            log_probs = logits.log_softmax(2)
            loss = criterion(log_probs, labels, input_lengths, label_lengths)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")

    # Save model
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\n✅ Model saved as: {MODEL_PATH}")

    # Evaluation on Train and Test
    print("\n📊 Evaluating model precision...")
    evaluate_precision(model, train_dataset, device, name="Train")

    test_dataset = RealOCRDataset('real_ocr_data/test/images', 'real_ocr_data/test/labels.txt')
    evaluate_precision(model, test_dataset, device, name="Test")
