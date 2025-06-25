import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
import string
import matplotlib.pyplot as plt
from datetime import datetime
import warnings

# -------- Config --------
warnings.filterwarnings("ignore", category=UserWarning)
CHARSET = string.ascii_letters + string.digits
CHAR2IDX = {c: i + 1 for i, c in enumerate(CHARSET)}
IDX2CHAR = {i + 1: c for i, c in enumerate(CHARSET)}
BLANK_IDX = 0
IMAGE_HEIGHT = 32
IMAGE_WIDTH = 160
MODEL_PATH = "real_ocr_data/crnn_model.pth"

# -------- Utils --------
def normalize(s): return s.strip().lower()

def encode_text(text):
    return [CHAR2IDX[c] for c in normalize(text) if c in CHAR2IDX]

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

def compute_precision(true_labels, pred_labels):
    tp = sum(1 for gt, pred in zip(true_labels, pred_labels) if normalize(gt) == normalize(pred))
    fp = sum(1 for gt, pred in zip(true_labels, pred_labels) if normalize(gt) != normalize(pred))
    precision = tp / (tp + fp + 1e-8)
    return tp, fp, precision

# -------- Dataset --------
class RealOCRDataset(Dataset):
    def __init__(self, image_folder, label_file, train=True):
        self.image_folder = image_folder
        self.samples = []
        with open(label_file, 'r') as f:
            for line in f:
                name, label = line.strip().split(maxsplit=1)
                self.samples.append((name, label))
        self.train = train
        self.train_transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
            transforms.RandomApply([
                transforms.RandomRotation(3),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
                transforms.GaussianBlur(3),
                transforms.RandomPerspective(distortion_scale=0.2, p=0.5)
            ], p=0.7),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
            transforms.ToTensor(),
        ])
        self.eval_transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
            transforms.ToTensor(),
        ])

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        filename, text = self.samples[idx]
        image_path = os.path.join(self.image_folder, filename)
        img = Image.open(image_path).convert('RGB')
        transform = self.train_transform if self.train else self.eval_transform
        img = transform(img)
        label = torch.tensor(encode_text(text), dtype=torch.long)
        return img, label, text

# -------- ResNet-CRNN Model --------
class ResNetCRNN(nn.Module):
    def __init__(self, num_classes):
        super(ResNetCRNN, self).__init__()
        resnet = models.resnet18(weights=None)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.cnn = nn.Sequential(*list(resnet.children())[:-3])
        dummy_input = torch.zeros(1, 1, IMAGE_HEIGHT, IMAGE_WIDTH)
        with torch.no_grad():
            cnn_out = self.cnn(dummy_input)
        _, c, h, _ = cnn_out.shape
        self.rnn_input_size = c * h
        self.rnn = nn.LSTM(
            input_size=self.rnn_input_size,
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            dropout=0.3,
            batch_first=True
        )
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        b, c, h, w = x.size()
        x = x.permute(0, 3, 1, 2).reshape(b, w, c * h)
        x, _ = self.rnn(x)
        x = self.dropout(x)
        x = self.fc(x)
        x = x.permute(1, 0, 2)
        return x

# -------- Training --------
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNetCRNN(num_classes=len(CHARSET) + 1).to(device)

    dataset = RealOCRDataset('real_ocr_data/train/images', 'real_ocr_data/train/labels.txt', train=True)
    total_size = len(dataset)
    test_size = int(0.2 * total_size)
    val_size = int(0.2 * test_size)
    test_size = test_size - val_size
    train_size = total_size - test_size - val_size
    train_set, test_set, val_set = random_split(dataset, [train_size, test_size, val_size])
    test_set.dataset.train = False
    val_set.dataset.train = False

    def collate_fn(batch):
        images, labels, raw_texts = zip(*batch)
        image_tensor = torch.stack(images)
        label_lengths = torch.tensor([len(l) for l in labels])
        labels = torch.cat(labels)
        with torch.no_grad():
            dummy_cnn_out = model.cnn(image_tensor[:1])
            output_width = dummy_cnn_out.shape[-1]
        input_lengths = torch.full((len(images),), output_width, dtype=torch.long)
        return image_tensor, labels, input_lengths, label_lengths, raw_texts

    train_loader = DataLoader(train_set, batch_size=8, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, collate_fn=collate_fn)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    criterion = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)

    def evaluate(dataloader, name):
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for images, _, _, _, raw_texts in dataloader:
                images = images.to(device)
                output = model(images).permute(1, 0, 2)
                preds = decode_output(output)
                y_true += list(raw_texts)
                y_pred += preds
        tp, fp, precision = compute_precision(y_true, y_pred)
        print(f"{name} Precision: TP={tp}, FP={fp}, Precision={precision:.4f}")
        return precision, tp, fp

    # --- Training Loop ---
    losses = []
    for epoch in range(100):
        model.train()
        total_loss = 0
        for images, labels, input_lengths, label_lengths, _ in train_loader:
            images, labels = images.to(device), labels.to(device)
            if torch.any(input_lengths < label_lengths): continue
            logits = model(images).log_softmax(2)
            try:
                loss = criterion(logits, labels, input_lengths, label_lengths)
            except:
                continue
            if torch.isnan(loss) or torch.isinf(loss): continue
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step(total_loss)
        print(f"\nEpoch {epoch + 1}, Loss: {total_loss:.4f}")
        losses.append(total_loss)
        val_precision, _, _ = evaluate(val_loader, "Validation")
        torch.save(model.state_dict(), MODEL_PATH)

    print("\n📊 Final Evaluation")
    model.load_state_dict(torch.load(MODEL_PATH))
    train_precision, tp_train, fp_train = evaluate(train_loader, "Train")
    test_precision, tp_test, fp_test = evaluate(test_loader, "Test")
    val_precision, tp_val, fp_val = evaluate(val_loader, "Validation")

    # ---- Save results ----
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save loss plot
    plt.figure()
    plt.plot(range(1, len(losses) + 1), losses, marker='o')
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    loss_img_path = f"results/loss_plot_{timestamp}.png"
    plt.savefig(loss_img_path)
    print(f"✅ Saved loss plot: {loss_img_path}")

    # Save results
    result_txt_path = f"results/result_summary_{timestamp}.txt"
    with open(result_txt_path, "w") as f:
        f.write("Final Evaluation Metrics\n")
        f.write(f"Train Precision: TP={tp_train}, FP={fp_train}, Precision={train_precision:.4f}\n")
        f.write(f"Test Precision: TP={tp_test}, FP={fp_test}, Precision={test_precision:.4f}\n")
        f.write(f"Validation Precision: TP={tp_val}, FP={fp_val}, Precision={val_precision:.4f}\n")
    print(f"✅ Saved results to: {result_txt_path}")
