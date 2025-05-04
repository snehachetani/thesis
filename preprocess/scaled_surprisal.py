import os
import torch
import pandas as pd
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

corpus_dir = 'data/raw-text-files/'


device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.to(device)
model.eval()
tokenizer.pad_token_id = tokenizer.eos_token_id


context_size = 20
T_list = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 8.0, 9.0, 10.0]


def get_scaled_logprobs(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    scaled_logits = logits / temperature
    probs = F.softmax(scaled_logits, dim=-1)
    surprisals = -1 * torch.log2(probs + 1e-9)
    return scaled_logits, probs, surprisals

# Read and process each text file
for filename in os.listdir(corpus_dir):
    if filename.endswith('.txt'):
        file_path = os.path.join(corpus_dir, filename)
        with open(file_path, 'r') as f:
            text = f.read()

        # Tokenize and prepare inputs
        #tokenized = tokenizer(" ".join(text))
        tokenized = tokenizer(text)
        batch_reshaped = {
            k: [[tokenizer.eos_token_id if k == "input_ids" else 1] + v[i:i+context_size-1] for i in range(0, len(v), context_size-1)] for k, v in tokenized.items()
        }

        # Pad the last sequence
        to_pad = context_size - len(batch_reshaped["input_ids"][-1])
        batch_reshaped["input_ids"][-1] = batch_reshaped["input_ids"][-1] + [tokenizer.pad_token_id for _ in range(to_pad)]
        batch_reshaped["attention_mask"][-1] = batch_reshaped["attention_mask"][-1] + [0 for _ in range(to_pad)]
        batch_reshaped["labels"] = [input_id if input_id != tokenizer.pad_token_id else -100 for input_id in batch_reshaped["input_ids"]]

        inputs = {filename.split('.')[0]: batch_reshaped}

        # Compute surprisals for each temperature
        for T in T_list:
            stories_scored = {}
            logits_all = []
            probs_all = []
            for item, batch in tqdm(inputs.items()):
                batch = {k: torch.LongTensor(v).to(device) for k, v in batch.items()}

                with torch.no_grad():
                    outputs = model(**batch)

                #token_surprisal = get_scaled_logprobs(outputs.logits, T)
                scaled_logits, probs, surprisals = get_scaled_logprobs(outputs.logits, T)
#
                scaled_logits_np = scaled_logits[:, :-1].cpu().numpy().reshape(-1, tokenizer.vocab_size)
                probs_np = probs[:, :-1].cpu().numpy().reshape(-1, tokenizer.vocab_size)
#
                # Ignore the last surprisal value of each context window
                token_surprisal = surprisals[:, :-1].cpu().numpy().reshape(-1, tokenizer.vocab_size)

                # Ignore the first label of each context window
                output_ids = batch["labels"][:, 1:].cpu().numpy().squeeze().reshape(-1)

                # Index for first dimension of surprisals
                index = torch.arange(0, output_ids.shape[0])
                token_surprisal = token_surprisal[index, output_ids]


                token_logits = scaled_logits_np[index, output_ids]  # Logits for input tokens
                token_probs = probs_np[index, output_ids]
                tokens = tokenizer.convert_ids_to_tokens(output_ids)
                assert len(tokens) == len(token_surprisal)

                stories_scored.update({item: (tokens, token_surprisal)})



                # Prepare the data for display
                all_data = []
                for idx, token in enumerate(tokens):
                    #token_id = output_ids[idx]
                    #token_logits = scaled_logits_np[idx]
                    #token_probs = probs_np[idx]
                    token_surprisal_value = token_surprisal[idx]

                    # Collect logits, probabilities, and surprisals for this token
                    all_data.append({
                        'Token': token,
                        'Logits': token_logits[idx].tolist(),
                        'Probabilities': token_probs[idx].tolist(),
                        'Surprisals': token_surprisal_value
                    })

                # Convert to DataFrame for better readability
                df = pd.DataFrame(all_data)

                # Display the DataFrame for the current temperature
                print(f"Results for temperature {T}:")
                print(df)  # Display token and surprisal for brevity
                print("\n")

            # Save results to a CSV file
            output_filename = f"{filename.split('.')[0]}_Temp_{T}.csv"
            output_dir = 'Thesis/Dundee/surprisal-values/'
            output_path = os.path.join(output_dir, output_filename)
            df.to_csv(output_path, index=False)
