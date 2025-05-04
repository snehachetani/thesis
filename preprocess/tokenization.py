def calculate_total_reading_time(data):
    # First, assign TRT as 0 for rows where FDUR = -99
    data.loc[data['SW_IDX'] == -99, 'TRT'] = 0
    
    # For rows where FDUR is not -99, calculate total reading time by summing FDUR for each WNUM
    data.loc[data['SW_IDX'] != -99, 'TRT'] = data.groupby('SW_IDX')['FDUR'].transform('sum')
    data['TRT'] = data['TRT'].astype(int)
    
    return data
            


def calculate_first_fixation_duration(data):
    data["FFD"] = 0  # Initialize FFD column
    max_seen_wnum = float("-inf")  # Track the max WNUM seen
    max_seen_subword = -1  # Track the max subword position seen for a given WNUM
    
    for i, row in data.iterrows():
        if row["SW_IDX"] == '-99':  # Ignore invalid rows (e.g., *Blink)
            continue
        
        # Extract WNUM and subword position from the IDX
        wnum, subword_pos = map(int, row["SW_IDX"].split('-'))  # Split IDX into WNUM and subword position
        
        if wnum > max_seen_wnum or (wnum == max_seen_wnum and subword_pos > max_seen_subword):
            # First valid fixation on a new word in proper order
            data.at[i, "FFD"] = row["FDUR"]
            
            # Update max_seen_wnum and max_seen_subword
            max_seen_wnum = wnum
            max_seen_subword = subword_pos
        else:
            # Regression case: word was skipped and revisited
            data.at[i, "FFD"] = 0

    return data


def calculate_first_pass_fixation_duration(data):
    data["FPFD"] = 0  # Initialize FPFD column
    max_seen_wnum = float("-inf")  # Track the max WNUM seen
    max_seen_subword = -1  # Track the max subword position seen for a given WNUM
    
    for i, row in data.iterrows():
        if row["SW_IDX"] == '-99':  # Skip invalid rows (e.g., *Blink)
            continue
        
        # Extract WNUM and subword position from the IDX
        try:
            wnum, subword_pos = map(int, row["SW_IDX"].split('-'))  # Split IDX into WNUM and subword position
        except ValueError:
            continue  # If IDX is not valid (e.g., '-99'), skip this row
        
        if wnum > max_seen_wnum or (wnum == max_seen_wnum and subword_pos > max_seen_subword):
            # First valid fixation on a new word in proper order
            gaze_duration = row["FDUR"]
            
            # Check for consecutive subwords with the same WNUM
            j = i + 1
            while j < len(data):
                next_row = data.iloc[j]
                if next_row["SW_IDX"] != '-99':  # Check if the next row is valid
                    next_wnum, next_subword_pos = map(int, next_row["SW_IDX"].split('-'))
                    if next_wnum == wnum and next_subword_pos == subword_pos:
                        # The next subword belongs to the same word (same WNUM), so we extend the gaze duration
                        gaze_duration += next_row["FDUR"]
                        data.at[j, "FPFD"] = gaze_duration  # Assign gaze duration to the next subword
                        #subword_pos += 1  # Move to the next subword
                        j += 1
                    else:
                        break
                else:
                    break
            
            # Assign the calculated gaze duration to the current word
            data.at[i, "FPFD"] = gaze_duration
            
            # Update max_seen_wnum and max_seen_subword
            max_seen_wnum = wnum
            max_seen_subword = subword_pos
        else:
            # Regression case: word was skipped and revisited
            data.at[i, "FPFD"] = 0

    return data
