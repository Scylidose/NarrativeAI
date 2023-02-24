import torch
from transformers import (GPTNeoForCausalLM, GPT2Tokenizer, TextDataset,
                          DataCollatorForLanguageModeling, Trainer,
                          TrainingArguments, PreTrainedTokenizer)
from torch.utils.data import ConcatDataset

import os
import glob
import random
import numpy as np


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


def train_gpt_neo(data_dir: str, output_dir: str,
                  model_name: str = "EleutherAI/gpt-neo-1.3B",
                  batch_size: int = 4, epochs: int = 1,
                  learning_rate: float = 2e-5,
                  weight_decay: float = 0.01):
    """
    Trains a GPT-Neo language model on the provided dataset.

    Args:
        data_dir (str): The directory containing the cleaned input data.
        output_dir (str): The directory to save the trained model and training
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
    tokenized_dataset = load_datasets(data_dir, tokenizer)

    data_collator = DataCollatorForLanguageModeling(
                        tokenizer=tokenizer,
                        mlm=False
                    )

    model = GPTNeoForCausalLM.from_pretrained(model_name)

    training_args = TrainingArguments(
        output_dir=output_dir,
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
        train_dataset=tokenized_dataset,
    )

    trainer.train()

    # Evaluate the model
    trainer.evaluate()

    # Save the model
    trainer.save_model(output_dir)
