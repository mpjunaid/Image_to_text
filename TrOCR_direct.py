import os
import pandas as pd
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# Load Pretrained TrOCR
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
model.eval()


# Inference Function
def predict_text_from_image(image_path):
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    predicted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return predicted_text


# Predict for CSV Images
def run_prediction(csv_file, image_subdir):
    base_dir = "cropped"
    image_dir = os.path.join(base_dir, "cropped", image_subdir)
    csv_path = os.path.join(base_dir, csv_file)

    df = pd.read_csv(csv_path)
    predictions = []

    for idx, row in df.iterrows():
        image_name = row["filename"]
        local_image_path = image_name.replace("/content/", "")
        # image_path = os.path.join(image_dir, image_name)

        try:
            pred_text = predict_text_from_image(local_image_path)
        except Exception as e:
            pred_text = f"[Error: {e}]"
        print(f"{image_name}: {pred_text}")
        predictions.append(pred_text)

    # Save output
    df["predicted_text"] = predictions
    output_csv = os.path.join(base_dir, "predictions_output.csv")
    df.to_csv(output_csv, index=False)
    print(f"\nPredictions saved to {output_csv}")


# Example usage
run_prediction("test_labels.csv", "Test")
