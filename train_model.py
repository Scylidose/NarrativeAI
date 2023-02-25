import torch
from transformers import (GPTNeoForCausalLM, GPT2Tokenizer, TextDataset,
                          DataCollatorForLanguageModeling, Trainer,
                          TrainingArguments, PreTrainedTokenizer)
from torch.utils.data import ConcatDataset

import os
import glob
import random
import numpy as np

import argparse


def load_dataset(file_path: str,
                 tokenizer: PreTrainedTokenizer) -> TextDataset:
    """
    Loads the text data from a file and encodes it using the provided
    tokenizer.

    Args:
        file_path (str): The path to the text file to be loaded.
        tokenizer (PreTrainedTokenizer): The tokenizer to be used to
                                         encode the text.

    Returns:
        TextDataset: The encoded dataset as a `TextDataset` object.
    """
    # Load the dataset
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=256
    )
    return dataset


def load_datasets(data_dir: str,
                  tokenizer: PreTrainedTokenizer) -> TextDataset:
    """
    Load all the dataset files and combine them into a single dataset

    Args:
        data_dir (str): The directory path where the text files are stored.
        tokenizer (PreTrainedTokenizer): The tokenizer to use for preprocessing
                                         the text.

    Returns:
        TextDataset: A concatenated dataset of all the preprocessed text files.
    """

    episode_datasets = []

    for season_dir in os.listdir(data_dir):
        if not os.path.isdir(os.path.join(data_dir, season_dir)):
            continue
        for episode_file in glob.glob(
            os.path.join(data_dir, season_dir, "episode_*.txt")
        ):
            episode_dataset = load_dataset(episode_file, tokenizer)
            episode_datasets.append(episode_dataset)

    return ConcatDataset(episode_datasets)


def set_seed(seed):
    """
    Set the random seed for reproducibility.

    Args:
        seed (int): The seed value to set.

    Returns:
        None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_gpt_neo(train_data_dir: str, eval_data_dir: str, model_dir: str,
                  model_name: str = "EleutherAI/gpt-neo-1.3B",
                  batch_size: int = 4, epochs: int = 1,
                  learning_rate: float = 2e-5,
                  weight_decay: float = 0.01):
    """
    Trains a GPT-Neo language model on the provided dataset.

    Args:
        train_data_dir (str): The directory containing the cleaned input
                              training data.
        eval_data_dir (str): The directory containing the cleaned input
                             eval data.
        model_dir (str): The directory to save the trained model and training
                         logs.
        model_name (str, optional): The name of the pretrained GPT-Neo
                                    model to use.
        batch_size (int, optional): The number of training samples per batch.
        epochs (int, optional): The number of times to iterate over the
                                training data.
        learning_rate (float, optional): The learning rate for the optimizer.
        weight_decay (float, optional): The amount of weight decay to apply
                                        during training.

    Returns:
        None
    """

    set_seed(42)

    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    train_tokenized_dataset = load_datasets(train_data_dir, tokenizer)
    eval_tokenized_dataset = load_datasets(eval_data_dir, tokenizer)

    data_collator = DataCollatorForLanguageModeling(
                        tokenizer=tokenizer,
                        mlm=False
                    )

    model = GPTNeoForCausalLM.from_pretrained(model_name)

    training_args = TrainingArguments(
        output_dir=model_dir,
        evaluation_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=weight_decay,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_tokenized_dataset,
        eval_dataset=eval_tokenized_dataset
    )

    trainer.train()

    # Evaluate the model
    trainer.evaluate()

    # Save the model
    trainer.save_model(model_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train a GPT-Neo model on input text data'
    )

    parser.add_argument(
        'train_data_dir', type=str,
        help='The directory containing the cleaned input training data.'
    )
    parser.add_argument(
        'eval_data_dir', type=str,
        help='The directory containing the cleaned input eval data.'
    )
    parser.add_argument(
        'model_dir', type=str,
        help='The directory to save the trained model and training logs.'
    )
    parser.add_argument(
        '--model_name', type=str, default="EleutherAI/gpt-neo-1.3B",
        help='The name of the pretrained GPT-Neo model to use.'
    )
    parser.add_argument(
        '--batch_size', type=int, default=4,
        help='The number of training samples per batch.'
    )
    parser.add_argument(
        '--epochs', type=int, default=1,
        help='The number of times to iterate over the training data.'
    )
    parser.add_argument(
        '--learning_rate', type=float, default=2e-5,
        help='The learning rate for the optimizer.'
    )
    parser.add_argument(
        '--weight_decay', type=float, default=0.01,
        help='The amount of weight decay to apply during training.'
    )

    args = parser.parse_args()

    train_gpt_neo(args.train_data_dir, args.eval_data_dir,
                  args.model_dir, args.model_name, args.batch_size,
                  args.epochs, args.learning_rate, args.weight_decay)
