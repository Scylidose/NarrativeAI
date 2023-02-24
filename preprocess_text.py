import re
import string
import contractions
import os
import glob


def preprocess_text(text: str) -> list:
    """
    Preprocesses the text by removing speaker annotations using
    regex, converting to lowercase, expanding contractions, removing
    newlines, special characters and punctuation, and tokenizing the text.

    Args:
        text (str): The text to be preprocessed.

    Returns:
        list: A list of tokens representing the preprocessed text.
    """
    # Remove speaker annotations using regex
    text = re.sub(r'\b[A-Z]+\b\s*\[[A-Z\s\.]+\]', '', text)

    # Convert to lowercase
    text = text.lower()

    # Expand contractions
    text = contractions.fix(text)

    # Remove newlines and punctuation
    text = text.replace('\n', ' ')

    # Remove special characters and punctuation
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)

    # Tokenize the text
    tokens = text.split()

    return tokens


def preprocess_dataset(raw_path: str):
    """
    Preprocesses a dataset of TV show transcripts by removing speaker
    annotations, converting to lowercase, expanding contractions, removing
    punctuation, and tokenizing the text. Writes the cleaned text to
    a new file in a separate directory.

    Args:
        raw_path (str): The path to the directory containing the raw
                        text files.

    Returns:
        None
    """
    # Iterate through data
    for season_dir in os.listdir(raw_path):
        if not os.path.isdir(os.path.join(raw_path, season_dir)):
            continue
        for episode_file in glob.glob(
            os.path.join(raw_path, season_dir, "*.txt")
        ):
            cleaned_path_parts = []
            cleaned_path = episode_file.split("/")

            # Change raw episode file path to redirect to cleaned episode file
            for path_part in cleaned_path:
                if 'raw' in path_part:
                    path_part = path_part.replace('raw', 'cleaned')
                    cleaned_path_parts.append(path_part)
                else:
                    cleaned_path_parts.append(path_part)

            cleaned_path = '/'.join(cleaned_path_parts)

            raw_file = open(episode_file, "r", encoding='utf-8')

            clean_file = open(cleaned_path, "w+", encoding='utf-8')

            # Preprocess every episode text to list of tokens
            clean_file.write(str(preprocess_text(raw_file.read())))
            clean_file.close()
            raw_file.close()
