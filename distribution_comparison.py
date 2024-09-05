import re
from collections import defaultdict
import numpy as np
from scipy.stats import entropy
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.feature_selection import mutual_info_classif

figures_folder = 'figures/'

def get_count_and_probabilities(leak):
    # Get leak file name
    leak = './' + leak + '/Stats.txt'
    # Open file and get stats
    with open(leak, 'r') as f:
        data = f.read()

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

def compute_kl_matrix(distributions, names):
    num_distributions = len(distributions)
    kl_matrix = np.zeros((num_distributions, num_distributions))

    for i in range(num_distributions):
        for j in range(num_distributions):
            if i != j:
                kl_matrix[i, j] = entropy(distributions[i], distributions[j])
            else:
                kl_matrix[i, j] = 0.0  # KL divergence between identical distributions is 0

    # Create a dataframe to better show data
    kl_df = pd.DataFrame(kl_matrix, index=names, columns=names)

    return kl_df

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

def plot_matrix(data, labels, cmap):
    # Create plot
    fig, ax = plt.subplots()
    cax = ax.matshow(data, cmap=cmap)

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
    distributions = []
    
    # For each leak get score distributions
    for leak in leak_names:
        count, probability = get_count_and_probabilities(leak)
        counts.append(count)
        distributions.append(probability)

    # Get the kl matrix
    kl_df = compute_kl_matrix(distributions, leak_names)

    # Plot and save the score distributions
    plot_distributions(distributions, leak_names).savefig(figures_folder + 'scores_distribution.png')
    # Plot and save the kl matrix
    plot_matrix(kl_df.values, leak_names, 'coolwarm').savefig(figures_folder + 'scores_kl_matrix.png')

if __name__ == '__main__':
    get_distribution_comparison()
    
    

    

            

