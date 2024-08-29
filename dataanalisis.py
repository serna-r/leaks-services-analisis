import pandas as pd
import math

verbose = 0
top_100_passwords = []

# Function to optimize password processing
def apply_all_in_one(password):
    mask, total = password_mask(password)
    simple_entropy = simple_entropy(password)
    shannon_entropy = shannon_entropy(password)
    length = len(password)
    load_top_100(file_path='top100.txt')
    common = is_common_password(password)

    return mask, total, simple_entropy, length, common

# Apply only one stat
def apply_one(password, metric):
    current_module = __import__(__name__)
    return getattr(current_module, metric)(password) 


def load_top_100(file_path):
    with open(file_path, 'r') as file:
        # Read all lines from the file, excluding the first line
        lines = file.readlines()[1:]  # Skip the first line
        
        # Limit the list to the top 100 passwords
        global top_100_passwords
        top_100_passwords = [line.strip() for line in lines[:100]]
    
    return top_100_passwords

def is_common_password(password):
    # Check if the password is in the list of the top 100
    return password in top_100_passwords

# Function to analyze password complexity
def password_mask(password):
    mask = ''
    if any(c.islower() for c in password): mask = mask + 'l'
    if any(c.isupper() for c in password): mask = mask + 'u'
    if any(c.isnumeric() for c in password): mask = mask + 'd'
    if any(not c.isalnum() for c in password): mask = mask + 's'
    total = True

    # For other languages
    if mask == '': mask = 'z'

    return mask, total

# Calculate simple_entropy using the formula E = L × log(R) / log(2)
def simple_entropy(word):
    word_length = len(word)
    possible_symbols = len(set(word))
    
    if possible_symbols > 1:  # To avoid log(1) or log(0)
        simple_entropy = word_length * (math.log(possible_symbols) / math.log(2))
    else:
        simple_entropy = 0  # If there is only one symbol, simple_entropy is 0
    
    return simple_entropy

def shannon_entropy(password):
    # Count the frequency of each character in the word
    frequencies = {}
    for char in password:
        if char in frequencies:
            frequencies[char] += 1
        else:
            frequencies[char] = 1
    
    # Calculate Shannon entropy
    entropy = 0.0
    word_length = len(password)
    for frequency in frequencies.values():
        probability = frequency / word_length
        entropy -= probability * math.log2(probability)
    
    return entropy

# Calculate only one stat
def one_stat(df, metric, output):
    # Apply stat to df
    df[metric] = df['Password'].apply(lambda x: apply_one(x, metric))
    if metric == 'shannon_entropy' or metric == 'simple_entropy':
        intervals = pd.cut(df[metric], bins=20)
        # Count how many passwords are in each interval
        interval_count = df.groupby(intervals, observed=True)['Password'].count()
        with open(output, 'w') as f:
            f.write(f'{interval_count.to_string()}')
    return

def statistics(df, output):
    # Show the first rows to ensure it loaded correctly
    if verbose > 0: print(df.head())

    # Generate a statistical report of the passwords
    # Count the most common passwords
    common_passwords = df['Password'].value_counts()

    # print("\nStatistical report of most common passwords:")
    if verbose > 0: print('common passwords:\n', common_passwords[:20])

    # Generate a statistical report of the passwords

    # Apply all the functions to each password and add to the dataframe
    df['mask'], df['total'], df['simple_entropy'], df['shannon_entropy'], df['length'], df['common'] = zip(*df['Password'].map(apply_all_in_one))

    if verbose >= 0: print('Applyed functions, now processing')

    # Count the frequency of each password length
    length_count = df['length'].value_counts().sort_index()
    # Show the result
    if verbose > 0: print(length_count)

    # Group by Character Length (for 6, 7, 8, 9, 10) and 'other' for longer or shorter passwords
    bins = [-float('inf'), 5, 6, 7, 8, 9, 10, float('inf')]
    labels = ['smaller','6', '7', '8', '9', '10', 'bigger']
    df['Length Group'] = pd.cut(df['length'], bins=bins, labels=labels)
    # Print to check
    if verbose > 1: print(df)
    # Aggregate the data
    table = df.groupby('Length Group', observed=True)['mask'].value_counts().unstack(level=1)
    table['total'] = df.groupby('Length Group', observed=True)['total'].sum()
    table = table.div(table['total'], axis=0).mul(100, axis=0)
    table = table.reindex(sorted(table.columns), axis=1)
    if verbose > 0: print(table)

    # Group the simple_entropy values into 20 intervals
    intervalsse = pd.cut(df['simple_entropy'], bins=20)
    # Count how many passwords are in each interval
    intervalse_count = df.groupby(intervalsse, observed=True)['Password'].count()
    # Show the table with the results
    if verbose > 0: print(intervalse_count)

    # Group the shannon_entropy values into 20 intervals
    intervalsshannon= pd.cut(df['shannon_entropy'], bins=20)
    # Count how many passwords are in each interval
    intervalshannon_count = df.groupby(intervalsshannon, observed=True)['Password'].count()
    # Show the table with the results
    if verbose > 0: print(intervalshannon_count)

    if verbose > 0: print('Password is common: ', df.groupby('common', observed=True).size().reset_index(name='count'))

    if verbose >= 0: print('\nData processed columns created: ', df.columns.values)

    # Open output file
    f = open(output, 'w')
    # Write data
    f.write(f"Total users read {len(df.index)} \n20 most common passwords \n{common_passwords[:20]} \nLength: \n{length_count.to_string()} \ntable \n{table.to_string()} \n{intervalse_count.to_string()}\n{intervalshannon_count.to_string()}\n Password is common:\n{df.groupby('common', observed=True).size().reset_index(name='count')}")

if __name__ == '__main__':
    # Load the CSV file into a DataFrame
    df = pd.read_csv('extracted_data.csv')
    output = 'Stats.txt'

    statistics(df, output)
