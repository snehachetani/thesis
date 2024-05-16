

def preprocess_data(result_df, result_df2):
    #result_df = result_df[:7430]
    words = result_df['WORD']
    wnum = result_df["WNUM"]
    wlen = result_df["WLEN"]
    xpos = result_df['XPOS']
    wdlp = list(result_df['WDLP'])
    oblp = list(result_df['OBLP'])
    fdur = result_df['FDUR']
    WORDs = result_df2['WORD']
    FXNO = result_df2['FXNO']
    TXFR  = result_df2['TXFR']
    WNUM = result_df2['WNUM']
    Participant_ID =  result_df2['Participant ID'] 
    Text_File = result_df2["Text File"]
    participant_id =  result_df['Participant ID'] 
    text_file = result_df["Text File"]
    words_to_ignore = ['*Off-screen', '*Blink']
    nan_words = result_df[result_df['WORD'].isna()]
    words.fillna('0', inplace=True)
    WORDs.fillna('0', inplace=True)
    words_without_nan = result_df.dropna(subset=['WORD'])
    return words, wnum, wlen, xpos, wdlp, oblp, fdur, words_to_ignore, participant_id, text_file, WORDs, FXNO, TXFR, WNUM, Participant_ID, Text_File