import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
import os
import string
import json

from transformers import GPT2Tokenizer, GPT2Model
from wordfreq import word_frequency

from preprocess.tokenization import *
from preprocess.selected_variables import *
from preprocess.reading_times import *



directory = "data/Dundee/english/"
temp_surprisal_dir = "data/Dundee/surprisal-values/"


for filename in os.listdir(directory):
    if filename.endswith("ma1p.dat"):
        # Skip files ending with '11ma1p.dat'
        #if filename.endswith(("11ma1p.dat", "13ma1p.dat")):
         #   continue

        participant_id = filename[:2]
        text_file = filename[2:4]
        file_path = directory + str(participant_id + text_file) + "ma1p.dat"  # According to order of the events

        ma1p = pd.read_csv(file_path, sep='\s+', skiprows=1,
                           names=['WORD', 'TEXT', 'LINE', 'OLEN', 'WLEN', 'XPOS', 'WNUM', 'FDUR', 'OBLP', 'WDLP', 'LAUN', 'TXFR'],
                           encoding='windows-1252', na_filter=False)
        print(str(participant_id + text_file) + "ma1 length = ", len(ma1p))

        file_path = "data/Dundee/english/items/tx" + str(text_file) + "wrdp.dat"  # Words according to text file
        txwrdp = pd.read_csv(file_path, sep='\s+',
                             names=['WORD', 'File NUM', 'TEXT', 'LINE', 'POS_Line', 'POS_Screen', 'YY', 'OLEN', 'WLEN', 'PUNC', 'O_PUNC', 'C_PUNC', 'WNUM', 'TXFR'],
                             encoding='windows-1252', na_filter=False)
        print("tx" + str(text_file) + "wrdp length = ", len(txwrdp))

        surprisal_files = [f"{temp_surprisal_dir}{text_file}_Temp_{i}.csv" for i in
                           ["1.0", "1.1", "1.2", "1.3", "1.4", "1.5", "1.6", "1.7", "1.8", "1.9", 
                            "2.0", "2.25", "2.5", "2.75", "3.0", "3.25", "3.5", "4.0", "4.5", 
                            "5.0", "5.5", "6.0", "6.5", "7.0", "8.0", "9.0", "10.0"]]
        
        surprisal_dfs = [pd.read_csv(f) for f in surprisal_files]
        surprisal_dfs = [df[df['Token'] != '<|endoftext|>'] for df in surprisal_dfs]


        print("*" * 150)
        print("loading done")
        print("*" * 150)

        # Extracted features from the given text files
        words_ma1, text_ma1, line_ma1, olen_ma1, wlen_ma1, xpos_ma1, wnum_ma1, fdur_ma1, oblp_ma1, wdlp_ma1, laun_ma1, txfr_ma1, words_txw, text_txw, line_txw, olen_txw, wlen_txw, wnum_txw, txfr_txw, pos_screen_txw, words_to_ignore = preprocess_data(ma1p, txwrdp)

        # Preprocessing words (removing special characters)
        cleaned_text_list = remove_special_characters(words_ma1, words_to_ignore)                # For ma1p files
        cleaned_text_ma1 = [cleaned_text for cleaned_text, special_char in cleaned_text_list]    # without any spacial characters
        special_char_ma1 = [special_char for cleaned_text, special_char in cleaned_text_list]    # special characters only
        cleaned_text_list = remove_special_characters(words_txw, words_to_ignore)                # For txwrdp files
        cleaned_text_txw = [cleaned_text for cleaned_text, special_char in cleaned_text_list]    # without any spacial characters
        special_char_txw = [special_char for cleaned_text, special_char in cleaned_text_list]    # special characters only

        # Tokenize Words into subwords
        subwords_ma1, makeup_words_ma1 = tokenize_words_into_subwords(words_ma1, words_to_ignore)
        cleaned_subwords_ma1, cleaned_makeup_words_ma1 = tokenize_words_into_subwords(cleaned_text_ma1, words_to_ignore)
        print("Length of subwords: ",len(subwords_ma1))
        subwords_txw, makeup_words_txw = tokenize_words_into_subwords(words_txw, words_to_ignore)
        cleaned_subwords_txw, cleaned_makeup_words_txw = tokenize_words_into_subwords(cleaned_text_txw, words_to_ignore)
        print("*"*150)
        print("tokenization done")
        print("*"*150)

        # Number of subwords of a word, length of each subword and the position of each subword in the text file
        num_subwords_ma1, subword_len_ma1, subword_pos_ma1 = calculate_subword_info(subwords_ma1, wnum_ma1, words_to_ignore)
        cl_num_subwords_ma1, cl_subword_len_ma1, cl_subword_pos_ma1 = calculate_subword_info(cleaned_subwords_ma1, wnum_ma1, words_to_ignore)
        num_subwords_txw, subword_len_txw, subword_pos_txw = calculate_subword_info(subwords_txw, wnum_txw, words_to_ignore)
        cl_num_subwords_txw, cl_subword_len_txw, cl_subword_pos_txw = calculate_subword_info(cleaned_subwords_txw, wnum_txw, words_to_ignore)

        # Fixated character within a word, fixated subword of a word, position of the fixated character in the fixated subword, index of the fixated subword
        fixated_char_ma1, subpart_ma1, fixated_pos_ma1, subword_idx_ma1 = gaze_landed_on_subwords(subwords_ma1, oblp_ma1, wnum_ma1, words_to_ignore)
        cl_fixated_char_ma1, cl_subpart_ma1, cl_fixated_pos_ma1, cl_subword_idx_ma1 = gaze_landed_on_subwords(cleaned_subwords_ma1, wdlp_ma1, wnum_ma1, words_to_ignore)
        print("*"*150)
        print("subword info and gaze done")
        print("*"*150)

        # Creating DataFrame using ma1p files with the subwords information
        ma1 = {'WORD': words_ma1, 'SBW': subwords_ma1, 'SBWNUM': num_subwords_ma1, #'SWNUM': cl_num_subwords_ma1, 'WLEN': wlen_ma1, 'SWLEN': cl_subword_len_ma1,
        'SWOLEN': subword_len_ma1, 'SPCHAR': special_char_ma1,  'WNUM': wnum_ma1,
        'SWIDX': subword_pos_ma1, 'XPOS': xpos_ma1, 'OBLP': oblp_ma1, 'WDLP': wdlp_ma1, 'Fixated_SBW': subpart_ma1,
        'SWDLP': fixated_pos_ma1, 'Fixated_char': fixated_char_ma1, 'FDUR': fdur_ma1, 'SW_IDX': subword_idx_ma1,}
        ma1 = pd.DataFrame(ma1)

        pd.set_option('display.expand_frame_repr', False)
        print("*"*100)
        #print(ma1[:20])
        print("*"*100)
        print('length of ma1:',len(ma1))
        print("*"*100)
        print("ma1 dataframe created")
        print("*"*150)

        # Calculating TRT, FFD, FPFD using created ma1 dataframe (event occured words)
        #data = {'Subword': ma1['Fixated_SBW'], 'IDX': ma1['SW_IDX'], 'FDUR': ma1['FDUR']}
        data = {'WORD': ma1['WORD'],'Fixated_SBW': ma1['Fixated_SBW'], 'SW_IDX': ma1['SW_IDX'], 'FDUR': ma1['FDUR']}
        data = pd.DataFrame(data)
        #df_FFD = find_ffd(df)
        #df_TRT = find_trt(df)
        #df_FPFD = find_fpfd(df)
        #df_FFD['FPFD'] = df_FPFD["FPFD"]
        #df_FFD['TRT']= df_TRT['TRT']
        #df_FFD.loc[df_FFD['FFD'] == 0, 'FPFD'] = 0
        #df_FFD['FDUR']=df_FFD['FDUR'].astype(int)
        #print(df_FFD)
        #metrics_df = df_FFD
        data = calculate_total_reading_time(data)
        data = calculate_first_fixation_duration(data)
        data = calculate_first_pass_fixation_duration(data)
        print(len(data))
        #metrics_df = calculate_metrics(ma1)
        print("*"*150)
        print("TRT, FFD, FPFD calculated")

        # save a final ma1 file with every details
        #metrics_df = metrics_df[~metrics_df['Subword'].isin(["-99"])]
        #metrics_df.rename(columns={'Subword': 'Fixated_SBW', 'IDX': 'SW_IDX'}, inplace=True)
        #ma1 = ma1.merge(metrics_df, on=['Fixated_SBW', 'SW_IDX', 'FDUR'], how='left')
        #ma1_final = pd.merge(ma1, metrics_df, left_on=['Fixated_SBW', 'SW_IDX'], right_on=['Subword', 'IDX'], how='left')
        #ma1_final = ma1_final.drop(columns=['Subword', 'IDX', 'FDUR_y'])
        filtered_metrics_df = data
        print(len(filtered_metrics_df))
        #print(filtered_metrics_df.columns)
        filtered_metrics_df = filtered_metrics_df.drop_duplicates(subset=['WORD', 'Fixated_SBW', 'SW_IDX', 'FDUR'])
        ma1 = ma1.merge(filtered_metrics_df, on=['WORD','Fixated_SBW', 'SW_IDX', 'FDUR'], how='inner')
        print(len(ma1))
        file_path1 = 'data/Dundee/ma1_files/ma1_'+str(participant_id+text_file)+'.csv'
        ma1.to_csv(file_path1, index=False)
        print('saved file_'+str(participant_id+text_file))
       #print("+"*150)

        # For all  words of the text file
        txwrdp_dict = {'WORD': words_txw, 'SBW': subwords_txw,
        'SBWLEN': subword_len_txw, 'SBWIDX': subword_pos_txw, 'POS_Screen': pos_screen_txw,}
        txwrdp_df = pd.DataFrame(txwrdp_dict)

        # Flatten the dataframe
        flattened_rows = []
        for idx, row in txwrdp_df.iterrows():
            if len(row['SBW']) > 1:
                for i in range(len(row['SBW'])):
                    flattened_rows.append({
                        'WORD': row['WORD'],
                        'SBW': row['SBW'][i],
                        'SBWLEN': row['SBWLEN'][i],
                        'SBWIDX': row['SBWIDX'][i],
                        'POS_Screen': row['POS_Screen']
                    })
            else:
                flattened_rows.append({
                    'WORD': row['WORD'],
                    'SBW': row['SBW'][0],
                    'SBWLEN': row['SBWLEN'][0],
                    'SBWIDX': row['SBWIDX'][0],
                    'POS_Screen': row['POS_Screen']
                })

        # Long dataframe for each subword in a row
        variables = pd.DataFrame(flattened_rows)
        print("*"*100)
        #print(variables.head(20))
        print("*"*100)

        ## Tokenizer tokenizes words into subword and some subwords are only "Ġ".
        variables['Ġ_SBW'] = variables['SBW']
        variables['SBW'] = variables['SBW'].str.replace('Ġ', '')
        # adding TRT, FFD, FPFD from the metrics dataframe to the new dataframe merged_df according to subwords and their index
        merged_df = pd.merge(variables, filtered_metrics_df, left_on=['WORD','SBW', 'SBWIDX'], right_on=['WORD','Fixated_SBW', 'SW_IDX'], how='left')
        merged_df = merged_df.drop_duplicates(subset=['WORD', 'SBW', 'SBWIDX', 'FDUR', 'SBWLEN', 'POS_Screen'])
        merged_df['fixated'] = np.where(merged_df['SBWIDX'] == merged_df['SW_IDX'], 1, 0)
        #merged_df = pd.merge(variables, metrics_df, left_on=['SBW', 'SBWIDX'], right_on=['Fixated_SBW', 'SW_IDX'], how='left')
        #merged_df['fixated'] = np.where(merged_df['SBWIDX'] == merged_df['SW_IDX'], 1, 0)
        merged_df[['FFD', 'TRT', 'FPFD', 'FDUR']] = merged_df[['FFD', 'TRT', 'FPFD', 'FDUR']].fillna(0).astype(int)
        variables_df = merged_df[['WORD', 'SBW', 'SBWLEN', 'SBWIDX', 'POS_Screen', 'FDUR', 'FFD', 'FPFD', 'TRT', 'Ġ_SBW', 'fixated']]
        print("*"*150)
        print("length of final dataframe: ", len(variables_df))

                # Adding all Surprisal columns
       # variables_df.loc[:, 'Surprisal'] = surprisal_df['surprisal'].values
        surprisal_columns = [f'Surprisal_{i}' for i in
                             ["1.0", "1.1", "1.2", "1.3", "1.4", "1.5", "1.6", "1.7", "1.8", "1.9", 
                              "2.0", "2.25", "2.5", "2.75", "3.0", "3.25", "3.5", "4.0", "4.5", 
                              "5.0", "5.5", "6.0", "6.5", "7.0", "8.0", "9.0", "10.0"]]

        for i, df in enumerate(surprisal_dfs):
            variables_df[surprisal_columns[i]] = df['Surprisal'].values

        print("Surprisal columns have been added")


        #variables_df.loc[:, 'TokenID'] = surprisal_df['token_id'].values
        print("*"*150)
        print("surprisal column has been added")

        # To check if the subword in a row is only a special char
        def has_only_punctuation(s):
            return all(char in string.punctuation or char.isspace() for char in s)
        variables_df['is_punc'] = variables_df['SBW'].apply(lambda x: 1 if has_only_punctuation(x) else 0)

        # Adding subword freq according to each text file
        #sbw_frequency = variables_df['SBW'].value_counts()
        #variables_df['Freq'] = variables_df['SBW'].map(sbw_frequency)

        # Checking if the word is splitted or not 
        variables_df['is_split'] = variables_df.apply(lambda row: 0 if row['WORD'] == row['SBW'] else 1, axis=1)
        # column of participant and text file information
        variables_df['ParticipantID'] = participant_id
        variables_df['TextFile'] = text_file

        print("*"*100)
        #print(variables_df[:25])
        print('length of total subwords in a file: ',len(variables_df))
        print("*"*100)

         ## Tokenizer tokenizes words into subword and some subwords are only "Ġ". So replaced this with empty string above
         ## and here removing those rows.
        print('length of empty string: ', len(variables_df[variables_df['SBW'] == '']))
        variables_df = variables_df[variables_df['SBW'] != '']
        #ma1p['WORD'] = ma1p['WORD'].replace('', ' ')
        variables_df.reset_index(drop=True, inplace=True)
        variables_df['WordIdx'] = variables_df['SBWIDX'].apply(lambda x: x.split('-')[0]) #only the word position
        file_path = './subword_freq_wiki_gpt2 1.json'

        with open(file_path, 'r') as file:
            data = json.load(file)
            
        total_frequency = sum(data.values())
        relative_frequencies = {key: freq / total_frequency for key, freq in data.items()}
        variables_df['WordFreq'] = variables_df['WORD'].apply(lambda x: word_frequency(x, 'en'))
        variables_df['SubwFreq'] = variables_df['Ġ_SBW'].map(relative_frequencies)
        # Saving the file
        file_path = 'data/Dundee/input-files/metrics_'+str(participant_id+text_file)+'.csv'
        variables_df.to_csv(file_path, index=False)
        print('saved file_'+str(participant_id+text_file))
        print("+"*150)
