from gpt2.model import GPT, GPTConfig

from typing import Optional, Type
import typer
import yaml
import tiktoken

import torch
import torch.nn.functional as F

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def get_tokens_from_text(text, num_return_sequences=5):
    encoder = tiktoken.get_encoding("gpt2")
    tokens = encoder.encode(text)
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
    return tokens

def generate_tokens_from_model(token_tensor, model:Type[GPT]):
    while token_tensor.size(1) < model.max_seq_length:
        with torch.no_grad():
            logits = model(token_tensor)
            logits = logits[:, -1, :] #take logits for last token in sequence. #(B, vocab_size)

            probs = F.softmax(logits, dim=-1) #convert logits to probabilities over total vocabulary

            #remove all tokens under top-50, to eliminate low-probability tokens and keep generation coherent.
            topk_probs, topk_indecies = torch.topk(probs, 50, dim=-1) 
            #selects one index, with probability of selection given by topk_probs
            selected_indecies = torch.multinomial(topk_probs, 1) 
            #get tokens from indecies
            selected_tokens = torch.gather(topk_indecies, -1, selected_indecies) 
            #continue generation by appending selected token to sequence
            token_tensor = torch.cat((token_tensor, selected_tokens), dim=1)
        
    for i in range(5):
        tokens = token_tensor[i, :model.max_seq_length].tolist()
        encoder = tiktoken.get_encoding("gpt2")
        decoded_text = encoder.decode(tokens)
        print(">", decoded_text)
            

def main(
    config_path: Optional[str] = None
):  
    print("begin inference")
    if config_path is not None:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        gpt_model = GPT(config=GPTConfig(config))
    else:
        gpt_model = GPT(GPTConfig())

    gpt_model.eval()
    device = get_device()
    gpt_model.to(device)

    text = "Hello, I'm a language model,"
    tokens = get_tokens_from_text(text)
    x = tokens.to(device)
    # logits, loss = gpt_model(x)

    print("generating output")
    generate_tokens_from_model(x, gpt_model)

if __name__ == "__main__":
    typer.run(main)