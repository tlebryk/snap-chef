import pandas as pd
from PIL import Image
from datasets import Dataset, Features, Sequence, Value, Image
import argparse
from typing import Optional


def convert_csv_to_huggingface_format(
    csv_file_path: str,
    images_folder: str,
    output_file_path: str,
    system_message: Optional[str] = None,
    pred_col: str = "Instructions",
    dataset_size: Optional[int] = None,
):
    if not system_message:
        system_message = "Give recipe instructions to the food you see in the picture"
    # Read the CSV file
    df = pd.read_csv(csv_file_path)
    if dataset_size:
        df = df.head(dataset_size)
    # Initialize the list to store all conversations
    huggingface_dataset = []

    # Process each row in the CSV
    for _, row in df.iterrows():
        print(f"{images_folder}/{row['Image_Name']}.jpg")
        data_dict = {
            "system_message": system_message,
            "user_message": {"image": f"{images_folder}/{row['Image_Name']}.jpg"},
            "assistant_message": row[pred_col],
        }

        huggingface_dataset.append(data_dict)
    features = Features(
        {
            "system_message": Value("string"),
            "user_message": {"image": Image()},
            "assistant_message": Value("string"),
        }
    )
    dataset = Dataset.from_list(huggingface_dataset, features=features)

    dataset.save_to_disk(output_file_path)
    return huggingface_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert CSV file to Huggingface dataset format"
    )
    parser.add_argument(
        "-c",
        "--csv_file_path",
        type=str,
        help="Path to the CSV file containing the data",
    )
    parser.add_argument(
        "-i",
        "--images_folder",
        type=str,
        help="Path to the folder containing the images",
    )
    parser.add_argument(
        "-o",
        "--output_file_path",
        type=str,
        help="Path to save the output Huggingface dataset",
        default="./",
    )
    parser.add_argument(
        "-d",
        "--dataset_size",
        type=int,
        help="Number of rows to process from the CSV file",
        optional=True,
    )

    args = parser.parse_args()
    convert_csv_to_huggingface_format(
        args.csv_file_path, args.images_folder, args.output_file_path
    )
