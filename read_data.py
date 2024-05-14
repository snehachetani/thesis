import os
import pandas as pd


def read_data_ma1p(directory):
    dfs = []
    for filename in os.listdir(directory):
        if filename.endswith("ma1p.dat"):
            participant_id = filename[:2]
            text_file = filename[2:4]
            file_path = os.path.join(directory, filename)
            df = pd.read_csv(file_path, sep='\s+', skiprows=1,
                             names=['WORD', 'TEXT', 'LINE', 'OLEN', 'WLEN', 'XPOS', 'WNUM', 'FDUR', 'OBLP', 'WDLP', 'LAUN', 'TXFR'], 
                             encoding='windows-1252')
            df['Participant ID'] = participant_id
            df['Text File'] = text_file
            dfs.append(df)
    result_df = pd.concat(dfs, ignore_index=True)
    return result_df


def read_data_ma2p(directory):
    dfs = []
    for filename in os.listdir(directory):
        if filename.endswith("ma2p.dat"):
            participant_id = filename[:2]
            text_file = filename[2:4]
            file_path = os.path.join(directory, filename)
            df = pd.read_csv(file_path, sep='\s+', skiprows=1,
                             names=['WORD', 'TEXT', 'LINE', 'OLEN', 'WLEN', 'XPOS', 'WNUM', 'FDUR', 'OBLP', 'WDLP', 'FXNO', 'TXFR'], 
                             encoding='windows-1252')
            df['Participant ID'] = participant_id
            df['Text File'] = text_file
            dfs.append(df)
    result_df = pd.concat(dfs, ignore_index=True)
    return result_df