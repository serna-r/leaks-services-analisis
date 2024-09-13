import re
from collections import defaultdict
import numpy as np
from scipy.stats import entropy
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import to_rgba

leaks_file = 'leak_types.txt'
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

# Get diferent colors for each type of service
def get_colors(leak_types):
    # Base colors for each category using matplotlib colormaps
    base_colors = {
        'business': plt.get_cmap('Blues'),
        'digitaltool': plt.get_cmap('Greens'),
        'social': plt.get_cmap('Reds'),
        'shopping': plt.get_cmap('Purples')
    }

    # Dictionary to track the number of times a category appears
    category_count = {
        'business': 0,
        'digitaltool': 0,
        'social': 0,
        'shopping': 0
    }
    
    colors = []
        
        # For each line in the file
    for leak in leak_types:
        category = leak[-1]  # The category is the second item of the dict
        
        # Increment the count for this category to vary the brightness
        category_count[category] += 1
        
        # Get a color from the colormap with different brightness
        colormap = base_colors[category]
        color = colormap(category_count[category] / len([l for l in leak_types if category in l]))  # Normalized brightness level
        
        # Append the color
        colors.append(to_rgba(color))
    
    return colors
            

def plot_distributions(distributions, names, colors=None):
    num_distributions = len(distributions)
    num_categories = len(distributions[0])
    
    x = np.arange(num_categories)  # Category positions for x-axis
    bar_width = 0.8 / num_distributions
    
    plt.figure(figsize=(12, 6))
    
    for i, distribution in enumerate(distributions):
        # Plot the bar with the appropiate color if it is specified
        plt.bar(x + i * bar_width, distribution, bar_width, label=names[i], color=colors[i] if colors else None)
    
    plt.xlabel('Scores zxcvbn')
    plt.ylabel('Probability')
    plt.title('Probability Distributions')
    plt.xticks(x + bar_width * (num_distributions - 1) / 2, [f'Score {i}' for i in range(num_categories)])
    plt.legend()
    plt.tight_layout()
    return plt


def plot_scores_by_length(distributions, names, colors=None):
        # Create a list to store all flattened distributions
    flattened_data = []

    # Iterate over the distributions (DataFrames) and flatten them
    for distribution, name in zip(distributions, names):
        # Flatten the distribution using stack to avoid NaNs
        flattened = distribution.set_index('length').stack(dropna=True).reset_index()
        flattened.columns = ['Length Group', 'Score', 'Probability']
        flattened['Distribution'] = name  # Add a column to identify the distribution
        flattened_data.append(flattened)

    # Concatenate all flattened DataFrames
    all_data = pd.concat(flattened_data, ignore_index=True)

    # Convert Length Group and Score to strings to avoid issues with plotting
    all_data['Length Group'] = all_data['Length Group'].astype(str)
    all_data['Score'] = all_data['Score'].astype(str)

    # Create a unique label combining 'Length Group' and 'Score' for each data point
    all_data['Label'] = all_data['Length Group'] + ' - Score ' + all_data['Score']

    # Create a bar plot
    plt.figure(figsize=(15, 8))  # Adjust size for long bar plots
    bar_width = 0.8 / len(distributions)  # Adjust bar width

    # Get unique x positions for each combination of 'Length Group' and 'Score'
    x_labels = all_data['Label'].unique()
    x_positions = np.arange(len(x_labels))

    for i, name in enumerate(names):
        # Filter the data for each distribution
        distribution_data = all_data[all_data['Distribution'] == name]

        # Convert 'Probability' to numeric to ensure there are no issues with invalid types
        distribution_data['Probability'] = pd.to_numeric(distribution_data['Probability'], errors='coerce')

        # Get the x positions that match the filtered data
        indices = [np.where(x_labels == label)[0][0] for label in distribution_data['Label']]

        # Ensure that the indices are integers for the x positions and valid probabilities
        indices = np.array(indices, dtype=int)

        # Plot the bars for the current distribution
        plt.bar(indices + i * bar_width, distribution_data['Probability'].fillna(0), bar_width, label=name, color=colors[i] if colors else None)

    plt.xlabel('Length Group and Score')
    plt.ylabel('Probability')
    plt.title('Distributions by Length and Score')

    # Adjust the x-axis labels for readability
    plt.xticks(x_positions + bar_width * (len(distributions) - 1) / 2, x_labels, rotation=90, ha='right', fontsize=8)

    plt.legend()
    plt.tight_layout()
    return plt

def plot_matrix(data, labels, cmap, vmin=0, vmax=2):
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 10))
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
    # Get leaks with categories
    leak_types = []
    # Read the leak names from a text file
    with open(leaks_file, "r") as file:
        for line in  file.read().splitlines():
            parts = line.rsplit(maxsplit=1)  # Split by last space
            entry_name = parts[0].strip()   # The entry name
            category = parts[1].strip()     # The category (remaining part of the line)

            # Append entry and category as a tuple to entries list
            leak_types.append((entry_name, category))

    # Sort entries by category name
    leak_types.sort(key=lambda x: x[1])

    # Get names list
    leak_names = [leak_name for leak_name, _ in leak_types]
    print(leak_types)
    print(leak_names)
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
    kl_dfs_score_length = compute_kl_matrix_dfs(score_length_dataframes, leak_names, 'length')

    # Get the colors for the plots
    colors = get_colors(leak_types)

    # Plot and save the score distributions
    plot_distributions(score_distributions, leak_names, colors).savefig(figures_folder + 'scores_distribution.png')
    # Plot and save the score by length distribution
    plot_scores_by_length(score_length_dataframes, leak_names, colors).savefig(figures_folder + 'scores_length_distribution.png')
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
    
    

    

            

