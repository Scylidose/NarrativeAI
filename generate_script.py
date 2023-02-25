from transformers import GPTNeoForCausalLM, GPT2Tokenizer
import torch

import argparse


def generate_text(model_dir: str, results_dir: str,
                  prompt: list, max_length: int = 10000,
                  num_sequences: int = 1,
                  do_sample: bool = True):
    """
    Generates new text based on a given prompt using a trained
    transformer model.

    Args:
        model_dir (str): The path to the directory where the pre-trained
                         model is stored.
        results_dir (str): The path to the directory where the generated
                           text should be stored.
        prompt (str): The prompt to generate new text from.
        max_length (int, optional): The maximum number of tokens to generate.
                                    Defaults to 10000.
        num_sequences (int, optional): The number of independent sequences
                                       to generate for each prompt.
                                       Defaults to 1.
        do_sample (bool, optional): Whether to use sampling or greedy decoding
                                    to generate new text.
                                    Defaults to True.

    Returns:
        str: The generated text.
    """
    # Load the saved model and tokenizer
    model = GPTNeoForCausalLM.from_pretrained(model_dir)
    tokenizer = GPT2Tokenizer.from_pretrained(model_dir)

    # Encode the prompt with the tokenizer
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    # Set attention mask and pad token ID
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long)
    pad_token_id = tokenizer.eos_token_id

    output = model.generate(input_ids, max_length=max_length,
                            num_return_sequences=num_sequences,
                            do_sample=do_sample, attention_mask=attention_mask,
                            pad_token_id=pad_token_id)

    # Decode the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Export generated text to a file
    with open(results_dir + "generated_text.txt", "w", encoding='utf-8') as f:
        f.write(generated_text)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Generate new text based on a given prompt using a '
                    'trained transformer model'
    )

    parser.add_argument(
        'model_dir', type=str,
        help='Path to the directory where the pre-trained model is stored'
    )
    parser.add_argument(
        'results_dir', type=str,
        help='Path to the directory where the generated text should be stored'
    )
    parser.add_argument(
        'prompt', type=str,
        help='The prompt to generate new text from'
    )
    parser.add_argument(
        '--max_length', type=int, default=10000,
        help='The maximum number of tokens to generate'
    )
    parser.add_argument(
        '--num_sequences', type=int, default=1,
        help='The number of independent sequences to generate for each prompt'
    )
    parser.add_argument(
        '--do_sample', type=bool, default=True,
        help='Whether to use sampling or greedy decoding to generate new text'
    )

    args = parser.parse_args()

    generate_text(args.model_dir, args.results_dir, args.prompt,
                  args.max_length, args.num_sequences, args.do_sample)
