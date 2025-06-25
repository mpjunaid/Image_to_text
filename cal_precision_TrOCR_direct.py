import pandas as pd
import os

def calculate_and_save_exact_precision(csv_path, output_txt_path):
    df = pd.read_csv(csv_path)

    # Clean up predictions and ground truths
    df['text_clean'] = df['text'].astype(str).str.strip().str.lower()
    df['predicted_clean'] = df['predicted_text'].astype(str).str.strip().str.lower()

    correct = (df['text_clean'] == df['predicted_clean']).sum()
    total = len(df)
    precision = correct / total if total > 0 else 0

    result = f"Exact Match Precision: {precision:.4f} ({correct}/{total})"
    print(result)

    # Save to txt file
    os.makedirs(os.path.dirname(output_txt_path), exist_ok=True)
    with open(output_txt_path, "w") as f:
        f.write(result)

# Example usage
calculate_and_save_exact_precision(
    csv_path="cropped/predictions_output.csv",
    output_txt_path="cropped/TrOCR_direct_precision.txt"
)