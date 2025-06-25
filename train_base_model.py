import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import random
import string
import matplotlib.pyplot as plt
import pandas as pd

# -------- Config --------
CHARSET = string.ascii_lowercase + string.digits
CHAR2IDX = {c: i + 1 for i, c in enumerate(CHARSET)}  # 0 is reserved for CTC blank
IDX2CHAR = {i + 1: c for i, c in enumerate(CHARSET)}
BLANK_IDX = 0
IMAGE_HEIGHT = 32
MAX_TEXT_LEN = 10
RESULTS_DIR = "result_base_model"
TRAIN_SPLIT_RATIO = 0.8
MODEL_SAVE_NAME = "crnn_ocr_model.pth" # Name for the saved model

# -------- Utils --------
def generate_image(text, width=100, height=32):
    img = Image.new('L', (width, height), color=255)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 20)  # Try to use a common font
    except IOError:
        font = ImageFont.load_default()
    draw.text((5, 5), text, font=font, fill=0)
    return img

def encode_text(text):
    return [CHAR2IDX[c] for c in text]

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

def calculate_precision(predictions, ground_truths):
    correct_predictions = 0
    for pred, gt in zip(predictions, ground_truths):
        if pred == gt:
            correct_predictions += 1
    return correct_predictions / len(predictions) if len(predictions) > 0 else 0

# -------- Dataset --------
class SyntheticOCRDataset(Dataset):
    def __init__(self, size=1000):
        self.data = []
        for _ in range(size):
            text = ''.join(random.choices(CHARSET, k=random.randint(3, MAX_TEXT_LEN)))
            img = generate_image(text)
            self.data.append((img, text))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, text = self.data[idx]
        img = img.resize((100, IMAGE_HEIGHT))
        img = np.array(img, dtype=np.float32) / 255.0
        img = torch.tensor(img).unsqueeze(0)  # [1, H, W]
        label = torch.tensor(encode_text(text), dtype=torch.long)
        return img, label, text # Return original text for evaluation

def collate_fn(batch):
    images, labels, texts = zip(*batch) # Unpack texts here
    image_tensor = torch.stack(images)
    label_lengths = torch.tensor([len(l) for l in labels])
    labels = torch.cat(labels)
    input_lengths = torch.full((len(images),), image_tensor.size(3) // 4, dtype=torch.long)
    return image_tensor, labels, input_lengths, label_lengths, texts

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
        x = self.cnn(x)  # [B, C, H, W]
        b, c, h, w = x.size()
        x = x.permute(0, 3, 1, 2)  # [B, W, C, H]
        x = x.reshape(b, w, c * h)  # [B, W, C*H]
        x, _ = self.rnn(x)
        x = self.fc(x)  # [B, W, num_classes]
        x = x.permute(1, 0, 2)  # [T, B, C] for CTC
        return x

# -------- Training Loop --------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create dataset and split into train and test
full_dataset = SyntheticOCRDataset(2000)
train_size = int(TRAIN_SPLIT_RATIO * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

model = CRNN(num_classes=len(CHARSET) + 1).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CTCLoss(blank=BLANK_IDX)

train_losses = []
print("Starting training...")
for epoch in range(10):
    model.train()
    total_loss = 0
    for images, labels, input_lengths, label_lengths, _ in train_loader: # Ignore texts for training loss
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)  # [T, B, C]
        log_probs = logits.log_softmax(2)
        loss = criterion(log_probs, labels, input_lengths, label_lengths)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_train_loss = total_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    print(f"Epoch {epoch + 1}, Loss: {avg_train_loss:.3f}")

# Create results directory if it doesn't exist
os.makedirs(RESULTS_DIR, exist_ok=True)

# -------- Save the trained model --------
model_save_path = os.path.join(RESULTS_DIR, MODEL_SAVE_NAME)
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

# -------- Evaluation and Saving Results --------
model.eval()

# Evaluate on training data
train_predictions = []
train_ground_truths = []
with torch.no_grad():
    for images, labels, input_lengths, label_lengths, texts in train_loader:
        images = images.to(device)
        logits = model(images)
        output = logits.permute(1, 0, 2) # [B, T, C]
        predicted_texts = decode_output(output)
        train_predictions.extend(predicted_texts)
        train_ground_truths.extend(texts)

train_precision = calculate_precision(train_predictions, train_ground_truths)
print(f"\nTraining Precision: {train_precision:.4f}")

# Evaluate on test data
test_predictions = []
test_ground_truths = []
with torch.no_grad():
    for images, labels, input_lengths, label_lengths, texts in test_loader:
        images = images.to(device)
        logits = model(images)
        output = logits.permute(1, 0, 2) # [B, T, C]
        predicted_texts = decode_output(output)
        test_predictions.extend(predicted_texts)
        test_ground_truths.extend(texts)

test_precision = calculate_precision(test_predictions, test_ground_truths)
print(f"Test Precision: {test_precision:.4f}")

# Save precision results
precision_filepath = os.path.join(RESULTS_DIR, "precision_results.txt")
with open(precision_filepath, "w") as f:
    f.write(f"Training Precision: {train_precision:.4f}\n")
    f.write(f"Test Precision: {test_precision:.4f}\n")
print(f"Precision results saved to {precision_filepath}")

# Save train predictions to CSV
train_df = pd.DataFrame({'Ground Truth': train_ground_truths, 'Predicted': train_predictions})
train_csv_filepath = os.path.join(RESULTS_DIR, "train_predictions.csv")
train_df.to_csv(train_csv_filepath, index=False)
print(f"Train predictions saved to {train_csv_filepath}")

# Save test predictions to CSV
test_df = pd.DataFrame({'Ground Truth': test_ground_truths, 'Predicted': test_predictions})
test_csv_filepath = os.path.join(RESULTS_DIR, "test_predictions.csv")
test_df.to_csv(test_csv_filepath, index=False)
print(f"Test predictions saved to {test_csv_filepath}")

# Plotting the loss graph
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o', linestyle='-', color='blue')
plt.title('Training Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.grid(True)
loss_plot_filepath = os.path.join(RESULTS_DIR, "training_loss.png")
plt.savefig(loss_plot_filepath)
print(f"Training loss plot saved to {loss_plot_filepath}")
