import re
from collections import defaultdict
import pandas as pd

def get_leak_types(leaks_file, dates=False):
    # Get leaks with categories
    leak_types = []
    if dates: dates_list = []
    # Read the leak names from a text file
    with open(leaks_file, "r") as file:
        for line in  file.read().splitlines():
            if not line.strip(): continue  # Skip empty lines
            elif line.startswith('#'): continue # Skip commented lines
            parts = line.rsplit(maxsplit=1)  # Split by last space
            date = parts[1].strip() # The last string is the date
            parts2 = parts[0].rsplit(maxsplit=1)
            entry_name = parts2[0].strip()   # The entry name
            category = parts2[1].strip()     # The category (remaining part of the line)

            # Append entry and category as a tuple to entries list
            leak_types.append((entry_name, category))
            # If dates requested
            if dates: 
                dates_list.append(date)

    # If dates needed return them
    if dates: 
        return leak_types, dates_list

    # Else retun only leak types
    return leak_types

def get_mask_distribution(data):
    
    # Get the file split in lines
    lines = data.splitlines()

    # Extract the table data from the content
    start_idx = None
    end_idx = None

    for i, line in enumerate(lines):
        if line.strip().startswith("Password mask:"):
            start_idx = i
        elif line.strip() == "":  # If empty line encountered, stop
            if start_idx is not None:
                end_idx = i
                break

    # Now extract the lines that correspond to the table
    table_lines = lines[start_idx:end_idx]

    # Create a DataFrame from the extracted lines
    # The first row is the header
    header = table_lines[1].split()
    data = []

    # Parse the remaining rows
    for line in table_lines[3:]:
        # # Eliminate nan ocurrences by substituting almost 0 not to break entropy
        # line = line.replace('NaN', '0.0')
        # Split in spaces
        row = line.split()

        # Convert value bigger and smaller to numbers
        if not row[0].isnumeric():
            if row[0] == 'smaller': row[0] = 0
            if row[0] == 'bigger': row[0] = -1

        # Append line
        data.append(map(float, row))

    # Create a pandas DataFrame
    df = pd.DataFrame(data, columns=header)

    # Drop column z to have same size matrices (this column only holds non utf-8 values)
    if 'z' in df.columns: df = df.drop(columns=['z'])
    
    # Eliminate the column total which is always 100, and return it
    return df.drop(columns=['total'])

def get_count(data):
    # Extract total users read
    total_users_match = re.search(r"Total users read: (\d+)", data)
    total_users = int(total_users_match.group(1)) if total_users_match else None

    return total_users

def get_count_and_probabilities(data):

    total_users = get_count(data)

    # Extract score distribution
    score_distribution = defaultdict(int)
    score_section_match = re.search(r"Score Distribution:\nscore\n((?:\d+\s+\d+\n)+)", data)

    if score_section_match:
        score_lines = score_section_match.group(1).strip().split('\n')
        for line in score_lines:
            score, count = map(int, line.split())
            score_distribution[score] = count

    # Convert scores into probabilities
    probability_dist = []
    count_list = []
    for score, count in score_distribution.items():
        count_list.append(count)
        probability_dist.append(count/total_users)

    # Return probability list
    return count_list, probability_dist

def get_score_and_length(file_path):
    # Read the file into a DataFrame
    df = pd.read_csv(file_path, sep='\s+', skiprows=1)

    # Rename the 'Length' column from the index to make it a column
    df.rename(columns={'score': 'length'}, inplace=True)
    # Drop the first row (which contains 'Length Group' etc.)
    df = df.drop(index=0).reset_index(drop=True)

    # Reset index so "Length" is no longer part of the index
    df.reset_index(drop=True, inplace=True)

    # Cast to float and get probabilities
    df[['0','1','2','3','4']] = df[['0','1','2','3','4']].astype(float)
    df[['0','1','2','3','4']] = df[['0','1','2','3','4']].div(df[['0','1','2','3','4']].sum(axis=1), axis=0)

    # Return the obtained df
    return df

def get_password_length_mean(data):
    # Find the Password Length Distribution section
    length_dist_section = re.search(r"Password Length Distribution:\nlength\n((?:\d+\s+\d+\n)+)", data)
    
    if length_dist_section:
        length_lines = length_dist_section.group(1).strip().split("\n")
        lengths = []
        counts = []
        for line in length_lines:
            length, count = map(int, line.split())
            lengths.append(length)
            counts.append(count)
        
        # Create a DataFrame
        df = pd.DataFrame({"Length": lengths, "Count": counts})
        
        # Calculate the mean length
        mean_length = (df["Length"] * df["Count"]).sum() / df["Count"].sum()
        return mean_length
    else:
        return None