import os

from generate_script import generate_text
from train_model import train_gpt_neo
from preprocess_text import preprocess_dataset


def main():
    """
    Main function to preprocess the Person of Interest
    dataset, train the GPT-Neo model, and generate sample text.

    Args:
        None

    Returns:
        None
    """
    # Set up file paths
    data_dir = "data/person_of_interest/"

    raw_path = os.path.join(data_dir, "raw")
    clean_path = os.path.join(data_dir, "cleaned")
    model_path = "models/"
    output_dir = "results/"
    prompt = "You are being watched. The government has a secret system:"

    # Preprocess data
    preprocess_dataset(raw_path)

    # Train GPT-Neo
    train_gpt_neo(clean_path, output_dir)

    # Generate new transcript
    generate_text(model_path, prompt)


if __name__ == "__main__":
    main()
