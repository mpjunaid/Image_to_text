import requests
from PIL import Image
# It's good practice to handle potential import errors
try:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
except ImportError:
    print("Error: The 'transformers' library is not installed.")
    print("Please install it by running: pip install transformers torch pillow")
    exit()

class TrOCRInference:
    """
    A class to handle OCR tasks using a pre-trained TrOCR model.
    The model is loaded once during initialization for efficient inference.
    """
    def __init__(self, model_name='microsoft/trocr-base-handwritten'):
        """
        Initializes and loads the TrOCR model and processor.
        """
        print("Initializing TrOCR model and processor...")
        try:
            self.processor = TrOCRProcessor.from_pretrained(model_name)
            self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
            print("...Model and processor loaded successfully.")
        except Exception as e:
            print(f"\n❌ An error occurred while loading the model from Hugging Face.")
            print("This could be a network issue or a problem with the transformers library.")
            print(f"Error details: {e}")
            # If the model fails to load, we cannot proceed.
            raise

    def process_image(self, image_path_or_url):
        """
        Performs OCR on a single image.

        Args:

            image_path_or_url (str): The path to a local image file or a URL of an image.

        Returns:
            str: The recognized text, or None if an error occurred.
        """
        # --- Stage 1: Load Image ---
        print(f"\nStage 1: Loading image from '{image_path_or_url}'...")
        try:
            if image_path_or_url.startswith(('http://', 'https://')):
                image = Image.open(requests.get(image_path_or_url, stream=True).raw).convert("RGB")
            else:
                print(f"Attempting to open image from local path: {image_path_or_url}")
                image = Image.open(image_path_or_url).convert("RGB")
            print("...Image loaded successfully.")
        except Exception as e:
            print(f"\n❌ An error occurred while opening the image.")
            print("Please check if the URL is correct or if the local file path exists.")
            print(f"Error details: {e}")
            return None

        # --- Stage 2: Process and Generate Text ---
        print("\nStage 2: Processing image and generating text (this may take a moment)...")
        try:
            print("Encoding image...")
            pixel_values = self.processor(images=image, return_tensors="pt").pixel_values
            print("Generating text...")
            generated_ids = self.model.generate(pixel_values)
            print("Decoding text...")
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            print("...Text generated successfully.")
        except Exception as e:
            print(f"\n❌ An error occurred during text generation or processing.")
            print(f"Error details: {e}")
            return None

        return generated_text


if __name__ == '__main__':
    try:
        # --- Stage 0: Initialize Model ---
        # This happens only once when the script starts.
        print("--- Starting OCR Process ---")
        ocr_engine = TrOCRInference()

        # Using the original, reliable URL for testing
        sample_image_url = r"D:\Python\Image_to_text\image.png"

        
        # You can switch to a local file by changing the line below:
        # target_image = 'path/to/your/image.png' 
        target_image = sample_image_url

        # --- Perform OCR ---
        recognized_text = ocr_engine.process_image(target_image)

        # --- Final Output ---
        if recognized_text is not None:
            print("\n" + "="*30)
            print(f"✅ Decoded Text: {recognized_text}")
            print("="*30)
        else:
            print("\n--- OCR process failed. See error messages above. ---")

    except Exception as e:
        # This will catch the error from __init__ if model loading fails.
        print(f"\nA critical error occurred during script execution: {e}")