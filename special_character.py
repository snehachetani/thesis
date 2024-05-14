
import re
import numpy as np
import pandas as pd
#words_to_ignore = ['*Off-screen', '*Blink']

def get_special_characters(words):
    special_chars = set()  # Using a set to avoid duplicates

    for word in words:
        # Find all non-alphanumeric characters in the word using regular expressions
        special_chars_in_word = re.findall(r'[^a-zA-Z0-9\s]', word)

        # Add the found special characters to the set
        special_chars.update(special_chars_in_word)

    return list(special_chars)

#cleaned_text_list = get_special_characters(words)
#print(cleaned_text_list)



def remove_special_characters(words, words_to_ignore):
    clean_text_list = []

    for word in words.values:  
        if pd.isna(word):  # if the word is NaN
            clean_text_list.append((np.nan, 0))  # Return 0 
        elif word in words_to_ignore:
            clean_text_list.append((word, '-99'))
        else:
            # Remove special characters within words except ' and - and \x9c
            cleaned_text = re.sub(r"(?<![\w\x9c'-])[^\w\s\x9c'-]+|[^\w\s\x9c'-](?![\w\x9c'-])|^'|'$", '', str(word)) 
            removed_chars = re.findall(r"(?<![\w\x9c'-])[^\w\s\x9c'-]+|[^\w\s\x9c'-](?![\w\x9c'-])|^'|'$", str(word))  
            
            # Append 0 if no special characters found, otherwise append the removed special characters
            clean_text_list.append((cleaned_text, 0 if not removed_chars else removed_chars))

    return clean_text_list
