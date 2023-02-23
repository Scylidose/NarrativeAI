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


### Usage

To train the language model, you can use the `train_model.py` script. This script takes as input the path to the transcripts directory and the model name to use, and outputs the trained model in the `models` directory. Here is an example of how to use the script:

```
python3 train_model.py --transcripts data/person_of_interest/raw --model-name <model_name>
```

To generate new transcripts using the trained model, you can use the `generate_script.py` script. This script takes as input the path to the model checkpoint and the length of the generated text, and outputs the generated text to the console. Here is an example of how to use the script:

```
python3 generate_script.py --checkpoint models/<model_name> --length 1000
```

### Folder Structure

The following is the recommended folder structure for this project:

```
├── data
│   └── person_of_interest
│       ├── raw
│       │   ├── season_1
│       │   │   ├── episode_1.txt
│       │   │   ├── episode_2.txt
│       │   │   ├── ...
│       │   ├── season_2
│       │   ├── ...
│       └── cleaned
│           ├── season_1
│           │   ├── episode_1.txt
│           │   ├── episode_2.txt
│           │   ├── ...
│           ├── season_2
│           ├── ...
├── models
│   
├── train_model.py
├── generate_script.py
├── preprocess_text.py
├── .gitignore
└── README.md
```

The `data` directory should contain the transcripts for the TV series, organized by season and episode. The `models` directory should contain the trained language models, organized by model name. The `preprocess_text.py` is used to preprocess the text applying diverse operations. The `train_model.py` and `generate_script.py` scripts are used for training and generating text, respectively. The `.gitignore` file specifies files and directories that should be ignored by Git. The `README.md` file is the documentation for the project.

## Contributing

Contributions to this project are welcome. If you find a bug, have a feature request, or want to contribute code, please open an issue or pull request on the project's GitHub page.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
