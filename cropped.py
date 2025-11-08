import os
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

def ocr_folder(folder_path):
    """
    Performs OCR on all images in a given folder.

    Args:
        folder_path (str): The path to the folder containing images.
    """
    # Load the processor and model
    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')

    # List all files in the directory
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        print(f"No images found in '{folder_path}'")
        return

    for img1_0 in image_files:
        image_path = os.path.join(folder_path, img1_0)
        try:
            image = Image.open(image_path).convert("RGB")
        except IOError:
            print(f"Could not open {image_path}. Skipping.")
            continue

        # Process the image and generate text
        pixel_values = processor(images=image, return_tensors="pt").pixel_values
        generated_ids = model.generate(pixel_values)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        print(f"File: {img1_0} -> Decoded Text: {generated_text}")


if __name__ == '__main__':
    # Replace 'path/to/your/cropped_images' with the actual folder path
    cropped_images_folder = r"D:\Python\Image_to_text\cropped\test"
    if os.path.isdir(cropped_images_folder):
         ocr_folder(cropped_images_folder)
    else:
        print(f"Directory not found: '{cropped_images_folder}'. Please update the path.")