import re
from collections import defaultdict
import numpy as np
from scipy.stats import entropy
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math

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
        # Eliminate nan ocurrences by substituting almost 0 not to break entropy
        line = line.replace('NaN', '0.0000000000000000000000000000000000000001')
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
    # Return the obtained df
    return df

def compute_kl_matrix(distributions, names):
    num_distributions = len(distributions)
    kl_matrix = np.zeros((num_distributions, num_distributions))

    for i in range(num_distributions):
        for j in range(num_distributions):
            if i != j:
                # print(distributions[i], distributions[j], '\n')
                kl_matrix[i, j] = entropy(distributions[i], distributions[j], nan_policy='omit')
                # if math.isinf(kl_matrix[i, j]): print(distributions[i], distributions[j])
            else:
                kl_matrix[i, j] = 0.0  # KL divergence between identical distributions is 0

    # Create a dataframe to better show data
    kl_df = pd.DataFrame(kl_matrix, index=names, columns=names)

    return kl_df

def compute_kl_matrix_dfs(distributions, names, dropcolumn, forceformat='float'):
    # Eliminate column without data
    distributions_drop_column = [df.drop([dropcolumn], axis=1) for df in distributions]

    kl_matrices = []
    # For each length distribution calculate kl matrix
    for i in range(len(distributions_drop_column[0].index)):
        # Get each length distribution
        length_dist = []
        for distribution in distributions_drop_column:
            if forceformat == 'none' : length_dist.append(distribution.iloc[i].to_list())
            # Check if there is a need to format the elements to float
            elif forceformat == 'float': length_dist.append(list(map(float, distribution.iloc[i].to_list())))
        
        # Append length and kl matrix
        kl_matrices.append([(i+5),compute_kl_matrix(length_dist, names)])

    return kl_matrices


def plot_distributions(distributions, names):
    num_distributions = len(distributions)
    num_categories = len(distributions[0])
    
    x = np.arange(num_categories)  # Category positions for x-axis
    bar_width = 0.10  # Width of each bar
    
    plt.figure(figsize=(12, 6))
    
    for i, distribution in enumerate(distributions):
        plt.bar(x + i * bar_width, distribution, bar_width, label=names[i])
    
    plt.xlabel('Scores zxcvbn')
    plt.ylabel('Probability')
    plt.title('Probability Distributions')
    plt.xticks(x + bar_width * (num_distributions - 1) / 2, [f'Score {i}' for i in range(num_categories)])
    plt.legend()
    plt.tight_layout()
    return plt

def plot_matrix(data, labels, cmap, vmin=0, vmax=2):
    # Create plot
    fig, ax = plt.subplots()
    # Set color scale to range from 0 to 2
    cax = ax.matshow(data, cmap=cmap, vmin=vmin, vmax=vmax)

    # Add color bar
    plt.colorbar(cax)

    # Set axis labels
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="left", rotation_mode="anchor")
    ax.set_yticklabels(labels)

    # Add top margin
    plt.subplots_adjust(top=0.8)

    # Annotate each cell with the numerical value
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f'{data[i, j]:.3f}', ha='center', va='center', color='white', fontsize=10)

    # Add white boxes to cover the diagonal
    for i in range(len(labels)):
        ax.add_patch(patches.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=True, color='white'))

    return plt

def get_distribution_comparison():
    # Read the leak names from a text file
    with open("leaks.txt", "r") as file:
        leak_names = file.read().splitlines()
    # Variable to store probability distributions
    counts = []
    score_distributions = []
    mask_dataframes = []
    score_length_dataframes = []
    
    # For each leak get score distributions
    for leak in leak_names:
        # Get leak file name
        leak_file = './' + leak + '/Stats.txt'
        # Open file and get stats
        with open(leak_file, 'r') as f:
            # Read file
            data = f.read()
            # Get count and probabilities for score
            count, probability = get_count_and_probabilities(data)
            # Get mask distributions
            mask_df = get_mask_distribution(data)

        # Define the path to the txt file
        leak_scores_and_length_file = leak + '\password_score_and_length.txt'
        score_length_df = get_score_and_length(leak_scores_and_length_file)

        # Get score counts, probabilities, scores and scores by length
        counts.append(count)
        score_distributions.append(probability)
        mask_dataframes.append(mask_df)
        score_length_dataframes.append(score_length_df)
    
    
    # Get the kl matrix for score
    kl_df_score = compute_kl_matrix(score_distributions, leak_names)

    # Get kl matrix for mask and length
    kl_dfs_mask = compute_kl_matrix_dfs(mask_dataframes, leak_names, 'mask')

    # Get kl matrix for score and length, format np to eliminate np.float values
    kl_dfs_score_length = compute_kl_matrix_dfs(score_length_dataframes, leak_names, 'length', forceformat='float')

    # Plot and save the score distributions
    plot_distributions(score_distributions, leak_names).savefig(figures_folder + 'scores_distribution.png')
    # Plot and save the kl matrix
    plot_matrix(kl_df_score.values, leak_names, 'coolwarm').savefig(figures_folder + 'scores_kl_matrix.png')

    # For each mask matrix plot
    for item in kl_dfs_mask:
        plot_matrix(item[1].values, leak_names, 'coolwarm').savefig(figures_folder + f'mask_length_{item[0]}_kl_matrix.png')

    # For each mask matrix plot
    for item in kl_dfs_score_length:
        plot_matrix(item[1].values, leak_names, 'coolwarm').savefig(figures_folder + f'score_length_{item[0]}_kl_matrix.png')

if __name__ == '__main__':
    get_distribution_comparison()
    
    

    

            

