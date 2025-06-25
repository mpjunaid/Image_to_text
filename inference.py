import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import string
import os

# -------- Config --------
CHARSET = string.ascii_lowercase + string.digits
CHAR2IDX = {c: i + 1 for i, c in enumerate(CHARSET)}  # 0 is reserved for CTC blank
IDX2CHAR = {i + 1: c for i, c in enumerate(CHARSET)}
BLANK_IDX = 0
IMAGE_HEIGHT = 32
MODEL_PATH = os.path.join("result_base_model", "crnn_ocr_model.pth")

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
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(b, w, c * h)
        x, _ = self.rnn(x)
        x = self.fc(x)
        x = x.permute(1, 0, 2)
        return x

# -------- Decode Output --------
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

# -------- Prediction Function --------
def predict_image(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CRNN(num_classes=len(CHARSET) + 1).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((IMAGE_HEIGHT, 100)),
        transforms.ToTensor(),
    ])

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        output = output.permute(1, 0, 2)
        pred_text = decode_output(output)[0]
    print("Hello")
    print("Predicted Text:", pred_text)
    return pred_text

# -------- CLI Test --------
if __name__ == "__main__":
    print("Hello")

    import sys
    if len(sys.argv) < 2:
        print("Usage: python inference.py path/to/image.png")
    else:
        predict_image(sys.argv[1])
