import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import to_rgba

# Get diferent colors for each type of service
def get_colors(leak_types):
    # Base colors for each category using matplotlib colormaps
    base_colors = {
        'business': plt.get_cmap('Blues'),
        'digitaltool': plt.get_cmap('Greens'),
        'shopping': plt.get_cmap('Greys'),
        'social': plt.get_cmap('Reds'),
        'games': plt.get_cmap('Purples')
    }

    # Dictionary to track the number of times a category appears
    category_count = {
        'business': 0,
        'digitaltool': 0,
        'shopping': 0,
        'social': 0,
        'games': 0
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
        distribution_data.loc[:, 'Probability'] = pd.to_numeric(distribution_data['Probability'], errors='coerce')

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
    # Get values without the main diagonal (as it is always o)
    non_diag_values = data[~np.eye(data.shape[0], dtype=bool)]

    # Get quantiles 0.10 and 0.15 if there is at least one value
    if len(non_diag_values) > 0:
        q025 = np.quantile(non_diag_values, .25)
        q010 = np.quantile(non_diag_values, .10)
        q015 = np.quantile(non_diag_values, .15)
    # Else al equal 0
    else:
        q025, q015, q010 = 0, 0, 0

    # Create plot
    fig_size = 10 + len(labels) * 0.2 if len(labels) > 10 else 10
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    # Set color scale to range from 0 to 2
    cax = ax.matshow(data, cmap=cmap, vmin=vmin, vmax=vmax)

    # Add color bar
    plt.colorbar(cax)

    # Set axis labels
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="left", rotation_mode="anchor", fontsize=min(12, 300 // len(labels)))
    ax.set_yticklabels(labels, fontsize=min(12, 300 // len(labels)))

    # # Add text for quantile explanation
    # plt.text(4, 10, f'Q1 (green) {q1:.5f}, Q0.15 (yellow) {q015:.5f}, Q0.10 (red) {q010:.5f}', horizontalalignment='center')
    ax.set_xlabel(f'Q0.25 (green) {q025:.5f}, Q0.15 (yellow) {q015:.5f}, Q0.10 (red) {q010:.5f}')

    # Add top margin
    plt.subplots_adjust(top=0.8)

    # Annotate each cell with the numerical value and add points if lower than quantile
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f'{data[i, j]:.3f}', ha='center', va='center', color='white', fontsize=10)
            if i!=j and data[i, j] < q010:
                ax.text(j, i+0.1, '.', ha='center', va='center', color='red', fontsize=20)
            elif i!=j and data[i, j] < q015:
                ax.text(j, i+0.1, '.', ha='center', va='center', color='yellow', fontsize=20)
            elif i!=j and data[i, j] < q025:
                ax.text(j, i+0.1, '.', ha='center', va='center', color='green', fontsize=15)
            

    # Add white boxes to cover the diagonal
    for i in range(len(labels)):
        ax.add_patch(patches.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=True, color='white'))

    return plt

# Unused
# Plot mask by length and score by length
def plot_by_length(leak_names, kl_dfs_mask, kl_dfs_score_length, figures_folder = 'figures/'):
    # For each mask length plot matrix
    for item in kl_dfs_mask:
        plot_matrix(item[1].values, leak_names, 'coolwarm').savefig(figures_folder + f'mask_length_{item[0]}_kl_matrix.png')

    # For each score and length plot the matrix
    for item in kl_dfs_score_length:
        plot_matrix(item[1].values, leak_names, 'coolwarm').savefig(figures_folder + f'score_length_{item[0]}_kl_matrix.png')


def boxwhiskers_from_kl_matrix(kl_matrix):
    # Get values without the main diagonal (as it is always o)
    non_diag_values = kl_matrix.to_numpy()[~np.eye(kl_matrix.shape[0], dtype=bool)]

    # Create boxplot
    plt.figure(figsize=(8, 6))
    plt.boxplot(non_diag_values, showmeans='True')
    plt.title('Box and Whiskers Plot (Ignoring Diagonal)')
    plt.ylabel('Values')
    plt.grid(True)

    # Show mean and quantiles
    q010 = np.quantile(non_diag_values, .10)
    q015 = np.quantile(non_diag_values, .15)
    q1 = np.quantile(non_diag_values, .25)
    q2 = np.quantile(non_diag_values, .50)
    q3 = np.quantile(non_diag_values, .75)
    plt.text(1, -0.2, f'mean: {np.mean(non_diag_values)}, Q1 {q1}, Q2 {q2}, Q3 {q3} \n Q0.10 {q010}, Q0.15 {q015}', horizontalalignment='center')
    
    return plt