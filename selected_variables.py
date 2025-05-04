

def preprocess_data(ma1p, txwrdp):
    ma1p['WORD'] = ma1p['WORD'].astype(str)
    print('length of negative wdlp value: ', len(ma1p[(ma1p['WDLP'] == -1) | (ma1p['WDLP'] == -2)]))

    ma1p = ma1p.drop(ma1p[(ma1p['WDLP'] == -1) | (ma1p['WDLP'] == -2)].index)
    #ma1p['WDLP'] = ma1p['WDLP'].replace([-1, -2], 0)
    #ma1p['WDLP'] = ma1p['WDLP'].replace(0, 1)
    #ma1p['OBLP'] = ma1p['OBLP'].replace(0, 1)

    print('length of empty string: ', len(ma1p[ma1p['WORD'] == '']))
    ma1p = ma1p[ma1p['WORD'] != '']
    #ma1p['WORD'] = ma1p['WORD'].replace('', ' ')

    ma1p.reset_index(drop=True, inplace=True)
    print('length after removing empty strings and negative wdlp words: ', len(ma1p))

    words_to_ignore = ['*Off-screen', '*Blink']

    words_ma1 = ma1p["WORD"]
    text_ma1 = ma1p["TEXT"]
    line_ma1 = ma1p["LINE"]
    olen_ma1 = ma1p['OLEN']
    wlen_ma1 = ma1p['WLEN']
    xpos_ma1 = ma1p['XPOS']
    wnum_ma1 = ma1p['WNUM']
    fdur_ma1 = ma1p['FDUR']
    oblp_ma1 = ma1p['OBLP']
    wdlp_ma1 = ma1p['WDLP']
    laun_ma1 = ma1p["LAUN"]
    txfr_ma1 = ma1p["TXFR"]

    words_txw = txwrdp["WORD"]
    text_txw = txwrdp["TEXT"]
    line_txw = txwrdp["LINE"]
    olen_txw = txwrdp['OLEN']
    wlen_txw = txwrdp['WLEN']
    wnum_txw = txwrdp['WNUM']
    txfr_txw = txwrdp["TXFR"]
    pos_screen_txw  = txwrdp['POS_Screen']

    return words_ma1, text_ma1, line_ma1, olen_ma1, wlen_ma1, xpos_ma1, wnum_ma1, fdur_ma1, oblp_ma1, wdlp_ma1, laun_ma1, txfr_ma1, words_txw, text_txw, line_txw, olen_txw, wlen_txw, wnum_txw, txfr_txw, pos_screen_txw, words_to_ignore
