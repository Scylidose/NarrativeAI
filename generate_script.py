from transformers import GPTNeoForCausalLM, GPT2Tokenizer
import torch


def generate_text(model_path: str, prompt: list,
                  max_length: int = 10000,
                  num_sequences: int = 1,
                  do_sample: bool = True):
    """
    Generates new text based on a given prompt using a trained
    transformer model.

    Args:
        model_path (str): The path to the directory where the pre-trained
                          model is stored.
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
    model = GPTNeoForCausalLM.from_pretrained(model_path)
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)

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
    with open("results/generated_text.txt", "w", encoding='utf-8') as f:
        f.write(generated_text)
