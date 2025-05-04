import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, AutoModelForCausalLM, AutoTokenizer
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()
    return model, tokenizer

if __name__ == "__main__":

    print("\n--- GPT2-small ---")
    gpt2_model, gpt2_tokenizer = load_model_and_tokenizer("gpt2")

    print("\n--- mGPT (Multilingual GPT) ---")
    mgpt_model, mgpt_tokenizer = load_model_and_tokenizer("ai-forever/mGPT")