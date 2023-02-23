import re
import string
import contractions
import os
import glob

# define a function to preprocess text
def preprocess_text(text):
    # remove speaker annotations using regex
    text = re.sub(r'\b[A-Z]+\b\s*\[[A-Z\s\.]+\]', '', text)
    
    # convert to lowercase
    text = text.lower()
        
    # Expand contractions
    text = contractions.fix(text)

    # Remove newlines and punctuation
    text = text.replace('\n', ' ')

    # remove special characters and punctuation
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    
    # tokenize the text
    tokens = text.split()
    
    return tokens


def preprocess_dataset(raw_path):

    # Iterate through data
    for season_dir in os.listdir(raw_path):
        if not os.path.isdir(os.path.join(raw_path, season_dir)):
            continue
        for episode_file in glob.glob(os.path.join(raw_path, season_dir, "*.txt")):

            # Change raw episode file path to redirect to cleaned episode file
            cleaned_path = episode_file.split("/")
            cleaned_path = '/'.join([path.replace('raw', 'cleaned') for path in cleaned_path])
            
            raw_file = open(episode_file, "r")

            clean_file = open(cleaned_path,"w+")

            # Preprocess every episode text to list of tokens
            clean_file.write(str(preprocess_text(raw_file.read())))
            clean_file.close()
            raw_file.close()


preprocess_dataset("data/person_of_interest/raw")