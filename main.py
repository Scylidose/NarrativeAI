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
    train_data_dir = "data/person_of_interest/train"
    train_raw_path = os.path.join(train_data_dir, "raw")
    train_clean_path = os.path.join(train_data_dir, "cleaned")

    eval_data_dir = "data/person_of_interest/eval"
    eval_raw_path = os.path.join(eval_data_dir, "raw")
    eval_clean_path = os.path.join(eval_data_dir, "cleaned")

    model_dir = "models/"
    results_dir = "results/"
    prompt = "You are being watched. The government has a secret system:"

    # Preprocess data
    preprocess_dataset(train_raw_path)
    preprocess_dataset(eval_raw_path)

    # Train GPT-Neo
    train_gpt_neo(train_clean_path, eval_clean_path, model_dir)

    # Generate new transcript
    generate_text(model_dir, results_dir, prompt)


if __name__ == "__main__":
    main()
