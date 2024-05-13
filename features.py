
# splitting words into subwords using GPT-small
def tokenize_words_into_subwords(words):
  subword_list = []
  makeup_word_list = []
  for word in words:
    if word in words_to_ignore:
      subword_list.append(word)
      makeup_word_list.append(word)
    else:
      tokens = tokenizer.tokenize(word)
      makeup_word = tokenizer.convert_tokens_to_string(tokens)
      #print(makeup_word)
      subword_list.append(tokens)
      makeup_word_list.append(makeup_word)

  return subword_list, makeup_word_list



 # number of sub-words  and the length of each sub-word
def calculate_subword_info(subwords, wnum):
  num_subwords = []
  subword_lengths = []
  subword_infos = []            # subword position within a subword ('un', 'believable') --> (1-1, 1-2)
  for sw in subwords:
    if sw in words_to_ignore:
      num_subwords.append(-99)
      subword_lengths.append(-99)
    else:
      num_subwords.append(len(sw))                    # number of subwords of a word
      subword_length = [len(sub) for sub in sw]       # length of each subword
      subword_lengths.append(subword_length)
    #print(subword_length)
  for i, sw in enumerate(subwords, 1):           # Enumerate over subwords - word and its index starting from 1
    if sw in words_to_ignore:
      subword_infos.append(-99)
    else:
      subword_info = [f"{wnum[i]}-{j}" for j in range(1, len(sw) + 1)]  # Creating subword info for each subword  1-1, 1-2
      subword_infos.append(subword_info)

  return num_subwords, subword_lengths, subword_infos



## Gaze landed within the fixated sub-word
def gaze_landed_on_subwords(cleaned_subwords, wdlp):
  char_fixated = []
  which_subpart = []
  within_subword = []

  for k, subword in enumerate(cleaned_subwords, 1):
    cumulative_length = 0
    zero_appended = False

    # Handling the case when subword is in words_to_ignore and wdlp is -99
    if subword in words_to_ignore and int(wdlp[k-1]) == -99:
      char_fixated.append(-99)
      which_subpart.append(-99)
      within_subword.append(-99)
    else:
      subword_char_fixated = []
      subword_which_subpart = []
      subword_within_subword = []

      for i, j in enumerate(subword, 1):
        offset = cumulative_length
        cumulative_length += len(j)

        if int(wdlp[k-1]) == 0 :              #and not zero_appended
          subword_char_fixated.append(0)
          subword_which_subpart.append(0)
          subword_within_subword.append(0)
          #zero_appended = True
        else:
          if int(wdlp[k-1]) <= cumulative_length:
            wdlp_value = int(wdlp[k-1]) - offset
            #print("wdlp_value:", wdlp_value)
            #print("j:", j)
            if wdlp_value <= len(j):          # and wdlp_value > 0
              sw = j[wdlp_value - 1 ]
              subword_char_fixated.append(sw)
              subword_which_subpart.append(j)
              subword_within_subword.append(wdlp_value)
              break
            else:
              print("Error: WDLP exceeds subword boundaries.")

      char_fixated.append(subword_char_fixated)
      which_subpart.append(subword_which_subpart)
      within_subword.append(subword_within_subword)


  return char_fixated, which_subpart, within_subword
