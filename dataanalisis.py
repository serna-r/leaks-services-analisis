import pandas as pd
import numpy as np
from parallelprocessing import parallel_proc_all, parallel_proc_one

verbose = 0
top_100_passwords = []

# Calculate only one stat
def one_stat(df, metric, output):
    # Apply stat to df
    df = parallel_proc_one(df, metric)
    if metric == 'shannon_entropy' or metric == 'simple_entropy':
        intervals = pd.cut(df[metric], bins=20)
        # Count how many passwords are in each interval
        interval_count = df.groupby(intervals, observed=True)['Password'].count()
        with open(output, 'w') as f:
            f.write(f'{interval_count.to_string()}')

    if metric == 'password_strength':
        # Common passwords
        common_passwords = df['Password'].value_counts()
        # Break column into score and guesses
        df[['score', 'guesses']] = pd.DataFrame(df['password_strength'].tolist(), index=df.index)

        # Process scores and count
        score_count = df.groupby('score', observed=True)['Password'].count()

        # # Create logarithmic bins
        # bins = np.logspace(np.log10(df['guesses'].min()), np.log10(df['guesses'].max()), 21)
        # # Apply pd.cut with these bins
        # intervalsguesses = pd.cut(df['guesses'], bins=bins)
        # # Count how many passwords are in each interval
        # intervalsguesses_count = df.groupby(intervalsguesses, observed=True)['Password'].count()

        # Write output
        with open(output, 'w') as f:
            f.write(f'Total users read: {len(df.index)}\n\n"20 Most Common Passwords:\n{common_passwords[:20]}\n\nScore Distribution:\n{score_count.to_string()}')

    if metric == 'password_score_and_length':
        df[['score', 'length']] = pd.DataFrame(df.password_score_and_length.tolist(), index= df.index)
        # Group by Character Length (for 6, 7, 8, 9, 10) and 'other' for longer or shorter passwords
        bins = [-float('inf'), 5, 6, 7, 8, 9, 10, float('inf')]
        labels = ['smaller','6', '7', '8', '9', '10', 'bigger']
        df['Length Group'] = pd.cut(df['length'], bins=bins, labels=labels)

        # Aggregate the data
        table = df.groupby('Length Group', observed=True)['score'].value_counts().unstack(level=1)
        # Write output
        with open(output, 'w') as f:
            f.write(f'Score by length\n{table.to_string()}')
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
    df = parallel_proc_all(df)

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

    # Print table with common passwords
    if verbose > 0: print('Password is common: ', df.groupby('common', observed=True).size().reset_index(name='count'))

    # Process scores and count
    score_count = df.groupby('score', observed=True)['Password'].count()
    if verbose > 0: print('Score count: ', score_count)

    # This causes bugs because of pandas checking for bins increasing monotonically
    try:
        # Group guesses for graphs
        # Create logarithmic bins
        bins = np.logspace(np.log10(df['guesses'].min()), np.log10(df['guesses'].max()), 21)
        # Apply pd.cut with these bins
        intervalsguesses = pd.cut(df['guesses'], bins=bins)
        # Count how many passwords are in each interval
        intervalsguesses_count = df.groupby(intervalsguesses, observed=True)['Password'].count()
    except Exception as e:
        # Create 
        intervalsguesses_count = pd.DataFrame()
        print(f"Exception:\n{e}\nBins\n{bins}")

    if verbose >= 0: print('\nData processed columns created: ', df.columns.values)

    # Open output file
    f = open(output, 'w')
    # Write data
    f.write(
    f"Total users read: {len(df.index)}\n\n"
    "20 Most Common Passwords:\n"
    f"{common_passwords[:20]}\n\n"
    "Password Length Distribution:\n"
    f"{length_count.to_string()}\n\n"
    "Password mask:\n"
    f"{table.to_string()}\n\n"
    "Simple entropy table:\n"
    f"{intervalse_count.to_string()}\n\n"
    "Shannon Entropy table:\n"
    f"{intervalshannon_count.to_string()}\n\n"
    "Password in most 100 most commmon:\n"
    f"{df.groupby('common', observed=True).size().reset_index(name='count')}\n\n"
    "Score Distribution:\n"
    f"{score_count}\n\n"
    "Guesses Interval Count:\n"
    f"{intervalsguesses_count.to_string()}\n"
    )

if __name__ == '__main__':
    # Load the CSV file into a DataFrame
    df = pd.read_csv('extracted_data.csv')
    output = 'Stats.txt'

    statistics(df, output)
