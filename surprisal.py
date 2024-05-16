import os
import pandas as pd
from transformers import GPT2Tokenizer, GPT2Model

"""
def read_txt(directory):
    dfs = []
    for filename in os.listdir(directory):
        if filename.endswith("wrdp.dat"):
            file_path = os.path.join(directory, filename)
            df = pd.read_csv(file_path, sep='\s+',
                             names=['WORD', 'Text File','SCREEN', 'LINE','POS','XX','YY', 'OLEN', 'WLEN', 'PUNC', 'O_PUNC', 'C_PUNC', 'WNUM', 'TXFR'], 
                             encoding='windows-1252', keep_default_na=False)
            dfs.append(df)
    txt = pd.concat(dfs, ignore_index=True)
    return txt

txt = read_txt("dundee_corpus/english/items/")

print(txt.isna().sum())
rows_with_nan = txt[txt['WORD'].isna()]

# Print the rows with NaN values in the 'WORD' column
print(rows_with_nan)


for text_file in txt['Text File'].unique():
    # Filter DataFrame for the current text file
    df_text_file = txt[txt['Text File'] == text_file]
    
    # Extract words and join them into a single string
    text = " ".join(df_text_file['WORD'])
    
    # Define the file path for the text file
    file_path = f"{text_file}.txt"
    
    # Write the text to the file
    with open(file_path, "w") as f:
        f.write(text)

        """


def process_files_surprisal(tokenizer, file_path_template, result_path_template):
    all_data = []

    # Process files from 1 to 20
    for i in range(1, 21):
        # file paths
        file_path = file_path_template.format(i)
        result_path = result_path_template.format(i)

        # word data file
        tx_df = pd.read_csv(file_path, sep='\s+', names=['WORD', 'Text File', 'SCREEN', 'LINE', 'POS', 'XX', 'YY', 'OLEN', 'WLEN', 'PUNC', 'O_PUNC', 'C_PUNC', 'WNUM', 'TXFR'], encoding='windows-1252', keep_default_na=False)

        # result file
        rs_df = pd.read_csv(result_path, sep="\t", keep_default_na=False)

        tx_df['Text File'] = i
        subword_list =[]

        # subwords
        txdf =[word if i == 0 else " " + word for i, word in enumerate(tx_df['WORD'])]
        for word in txdf:
            tokens = tokenizer.tokenize(word)
            subword_list.extend(tokens)

        # DataFrame for each file
        dict3 = {"subword": subword_list, "subw": rs_df['token'], "surprisal": rs_df['surprisal'], "Text File": i}
        surprisal_df = pd.DataFrame(dict3)

        # Append data to the list
        all_data.append(surprisal_df)

    # Combining into a single DataFrame
    combined_df = pd.concat(all_data, ignore_index=True)
    
    return combined_df
