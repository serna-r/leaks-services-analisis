import re
from collections import defaultdict
import numpy as np
from scipy.stats import entropy
import pandas as pd
import warnings
from plots import get_colors, plot_distributions, plot_matrix, plot_scores_by_length, boxwhiskers_from_kl_matrix

# Suppress the FutureWarning and the runtime one
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

figures_folder = 'figures/'

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

def get_count_and_probabilities(data):

    # Extract total users read
    total_users_match = re.search(r"Total users read: (\d+)", data)
    total_users = int(total_users_match.group(1)) if total_users_match else None

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

def compute_kl_matrix(distributions, names):
    num_distributions = len(distributions)
    kl_matrix = np.zeros((num_distributions, num_distributions))

    for i in range(num_distributions):
        for j in range(num_distributions):
            if i != j:
                kl_matrix[i, j] = entropy(distributions[i], distributions[j], nan_policy='omit')
            else:
                kl_matrix[i, j] = 0.0  # KL divergence between identical distributions is 0

    # Create a dataframe to better show data
    kl_df = pd.DataFrame(kl_matrix, index=names, columns=names)

    return kl_df

def compute_kl_matrix_dfs(distributions, names, dropcolumn):
    # Eliminate column without data
    distributions_drop_column = [df.drop([dropcolumn], axis=1) for df in distributions]

    kl_matrices = []
    # For each length distribution calculate kl matrix
    for i in range(len(distributions_drop_column[0].index)):
        # Get each length distribution
        length_dist = []
        for distribution in distributions_drop_column:
            length_dist.append(distribution.iloc[i].to_list())
        
        # Append length and kl matrix
        kl_matrices.append([(i+5),compute_kl_matrix(length_dist, names)])

    return kl_matrices


def get_distribution_comparison(leaks_file='leak_types.txt'):
    # Get leaks with categories
    leak_types = []
    # Read the leak names from a text file
    with open(leaks_file, "r") as file:
        for line in  file.read().splitlines():
            if not line.strip(): continue  # Skip empty lines
            elif line.startswith('#'): continue # Skip commented lines
            parts = line.rsplit(maxsplit=1)  # Split by last space
            entry_name = parts[0].strip()   # The entry name
            category = parts[1].strip()     # The category (remaining part of the line)

            # Append entry and category as a tuple to entries list
            leak_types.append((entry_name, category))

    # Sort entries by category name
    leak_types.sort(key=lambda x: x[1])

    # Get names list
    leak_names = [leak_name for leak_name, _ in leak_types]
  
    # Get unique categories
    leak_categories = set([leak_type for _, leak_type in leak_types])

    # Variable to store probability distributions
    counts = []
    score_distributions = []
    mask_dataframes = []
    score_length_dataframes = []
    
    # For each leak get score distributions
    for leak in leak_names:
        # Get leak file name
        leak_file = './leaks/' + leak + '/Stats.txt'
        # Open file and get stats
        with open(leak_file, 'r') as f:
            # Read file
            data = f.read()
            # Get count and probabilities for score
            count, probability = get_count_and_probabilities(data)
            # Get mask distributions
            mask_df = get_mask_distribution(data)

        # Define the path to the txt file
        leak_scores_and_length_file = 'leaks\\' + leak + '\password_score_and_length.txt'
        score_length_df = get_score_and_length(leak_scores_and_length_file)

        # Get score counts, probabilities, scores and scores by length
        counts.append(count)
        score_distributions.append(probability)
        mask_dataframes.append(mask_df)
        score_length_dataframes.append(score_length_df)
    
    # Get the kl matrix for score
    kl_df_score = compute_kl_matrix(score_distributions, leak_names)
    kl_df_score.to_csv('./leaks/kl_df_score.csv')

    # # Unused. Useful to plot by length scores and masks
    # # Get kl matrix for mask and length
    # kl_dfs_mask = compute_kl_matrix_dfs(mask_dataframes, leak_names, 'mask')

    # # Get kl matrix for score and length, format np to eliminate np.float values
    # kl_dfs_score_length = compute_kl_matrix_dfs(score_length_dataframes, leak_names, 'length')

    # Get the colors for the plots
    colors = get_colors(leak_types)

    # Plot and save the score distributions
    plot_distributions(score_distributions, leak_names, colors).savefig(figures_folder + 'scores_distribution.png')
    # Plot and save the score by length distribution
    plot_scores_by_length(score_length_dataframes, leak_names, colors).savefig(figures_folder + 'scores_length_distribution.png')
    # Plot and save the kl score matrix
    plot_matrix(kl_df_score.values, leak_names, 'coolwarm', vmin=0, vmax=1).savefig(figures_folder + 'scores_kl_matrix.png')
    # Get box and whiskers plot for values in the score kl matrix
    boxwhiskers_from_kl_matrix(kl_df_score).savefig(figures_folder + 'score_boxwhiskers_klmatrix.png')

    # Get stats for each category
    for category in leak_categories:
        # Get the leaks in the category
        leaks_in_category = [name for name, type in leak_types if type == category]
        # Get a dataframe with the kl values of the category
        category_df = kl_df_score[leaks_in_category].loc[leaks_in_category]
        # Plot category matrix
        plot_matrix(category_df.values, leaks_in_category, 'coolwarm', vmin=0, vmax=1).savefig(f'{figures_folder}/c_{category}_scores_kl_matrix.png')



if __name__ == '__main__':
    get_distribution_comparison()
    
    

    

            

