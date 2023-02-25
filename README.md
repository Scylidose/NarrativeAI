# NarrativeAI

NarrativeAI is a project that uses machine learning to generate new transcripts for the Person of Interest TV series. The project is based on the idea of training a language model on the existing transcripts of the show and using it to generate new ones.

## Getting Started

### Prerequisites

To run this project, you will need:

* Python 3.6 or higher
* The transcripts for the Person of Interest TV series (for training the language model)

### Installation and Setup

1. Clone this repository to your local machine.

```
git clone https://github.com/Scylidose/NarrativeAI.git
```

2. Create and activate a new virtual environment:

```
python3 -m venv env
source env/bin/activate # on Linux or macOS
.\env\Scripts\activate # on Windows
```

3. Install the required Python packages:

```
pip3 install -r requirements.txt
```

This will install all the necessary dependencies, including `transformers`, which provides an interface for using the pre-trained GPT-NEO model.

Now that you have set up your environment, you can proceed to running the scripts to generate new transcripts using the pre-trained model.

#### Notes 

- I use the GPT-NEO model instead of the newer ones for cost, availability and complexity reasons.


## Usage

### Train the model

#### Description

To train the language model, you can use the `train_model.py` script. This script trains a GPT-Neo model on input text data and saves the trained model to a specified directory.

```
python3 train_model.py [-h] [--model_name MODEL_NAME] [--batch_size BATCH_SIZE]
                      [--epochs EPOCHS] [--learning_rate LEARNING_RATE]
                      [--weight_decay WEIGHT_DECAY]
                      train_data_dir eval_data_dir model_dir
```

#### Arguments

- `train_data_dir`: The directory containing the cleaned input training data.
- `eval_data_dir`: The directory containing the cleaned input eval data.
- `model_dir`: The directory to save the trained model and training logs.
- `--model_name`: The name of the pretrained GPT-Neo model to use. Default is EleutherAI/gpt-neo-1.3B.
- `--batch_size`: The number of training samples per batch. Default is 4.
- `--epochs`: The number of times to iterate over the training data. Default is 1.
- `--learning_rate`: The learning rate for the optimizer. Default is 2e-5.
- `--weight_decay`: The amount of weight decay to apply during training. Default is 0.01.

#### Example

```
python3 train_model.py data/person_of_interest/train/ data/person_of_interest/eval/ models/
```

### Generate Script

#### Description 

To generate new transcripts using the trained model, you can use the `generate_script.py` script. This script generates new text based on a given prompt using a pre-trained transformer model. The generated text is saved to a specified directory.

```
python3 generate_script.py [-h] [--max_length MAX_LENGTH]
                          [--num_sequences NUM_SEQUENCES] [--do_sample DO_SAMPLE]
                          model_dir results_dir prompt [prompt ...]
```

#### Arguments
- `model_dir`: Path to the directory where the pre-trained model is stored.
- `results_dir`: Path to the directory where the generated text should be stored.
- `prompt`: The prompt to generate new text from.
- `--max_length`: The maximum number of tokens to generate. Default is 10000.
- `--num_sequences`: The number of independent sequences to generate for each prompt. Default is 1.
- `--do_sample`: Whether to use sampling or greedy decoding to generate new text. Default is True.

#### Example

```
python3 generate_script.py models/ results/ "You are being watched." --num_sequences 100
```

### All together 

To execute all the step (preprocessing, training and generating) in one command you can simply use the script:

```
python3 main.py
```

### Folder Structure

The following is the recommended folder structure for this project:

```
├── data
│   └── person_of_interest
│       ├── train
│       │   ├── raw
│       │   │   ├── season_1
│       │   │   │   ├── episode_1.txt
│       │   │   │   ├── ...
│       │   │   │── season_2
│       │   │   │── ...
│       │   └── cleaned
│       │       ├── season_1
│       │       │   ├── episode_20.txt
│       │       │   ├── ...
│       │       │── season_2
│       │       │── ...
│       │   
│       └── eval
│           ├── raw
│           │   ├── season_1
│           │   │   ├── episode_20.txt
│           │   │   ├── ...
│           │   │── season_2
│           │   │── ...
│           └── cleaned
│               ├── season_1
│               │   ├── episode_20.txt
│               │   ├── ...
│               │── season_2
│               │── ...
├── models
│
├── results
│
├── train_model.py
├── generate_script.py
├── preprocess_text.py
├── .gitignore
├── requirements.txt
└── README.md
```

The `data` directory should contain the transcripts for the TV series, organized by season and episode.  
The `models` directory should contain the trained language models, organized by model name.  
The `results` directory should contain the generated text, outputed by the trained and saved model.  
The `preprocess_text.py` is used to preprocess the text applying diverse operations  
The `train_model.py` and `generate_script.py` scripts are used for training and generating text, respectively.  
The `.gitignore` file specifies files and directories that should be ignored by Git.  
The `requirements.txt` file list all the necessary packages to run the project.  
The `README.md` file is the documentation for the project.

## Contributing

Contributions to this project are welcome. If you find a bug, have a feature request, or want to contribute code, please open an issue or pull request on the project's GitHub page.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
