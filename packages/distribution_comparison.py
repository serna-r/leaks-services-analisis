import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import entropy, kruskal
import pandas as pd
import warnings
from packages.retrieve_stats import get_count_and_probabilities, get_mask_distribution, get_score_and_length, get_leak_types
from packages.plots import get_colors, plot_distributions, plot_matrix, plot_scores_by_length, boxwhiskers_from_kl_matrix, plot_by_length, boxwhiskers_from_kl_matrices, random_scatterplot_klmatrices, plot_by_year, plot_by_year_average

# Suppress the FutureWarning and the runtime one
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

# Figures storage
FIGURES_FOLDER = 'figures/leaks/'

# Boolean to choose analysis by length
LENGTH = False

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

def kruskal_wallis_distance(distributions, names):

    # Realizar el test de Kruskal-Wallis
    h_stat, p_value = kruskal(*distributions)
    
    # Resultado
    result = {
        "H_statistic": h_stat,
        "p_value": p_value,
        "groups": names
    }
    return result


def get_distribution_comparison(leaks_file='leak_types.txt'):
    # Get leaks with categories
    leak_types, dates_list = get_leak_types(leaks_file, dates=True)

    # Sort entries by category name
    leak_types.sort(key=lambda x: x[1])

    # Get names list
    leak_names = [leak_name for leak_name, _ in leak_types]
  
    # Get unique categories and order them
    leak_categories = list(set([leak_type for _, leak_type in leak_types]))
    leak_categories.sort()

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
            count, probability = get_count_and_probabilities(leak)
            # Get mask distributions
            mask_df = get_mask_distribution(data)

        # Only execute if score by length is needed
        # Define the path to the txt file
        if LENGTH:
            leak_scores_and_length_file = 'leaks\\' + leak + '\password_score_and_length.txt'
            score_length_df = get_score_and_length(leak_scores_and_length_file)
            score_length_dataframes.append(score_length_df)

        # Get score counts, probabilities, scores and scores by length
        counts.append(count)
        score_distributions.append(probability)
        mask_dataframes.append(mask_df)
    
    # Get the kl matrix for score
    kl_df_score = compute_kl_matrix(score_distributions, leak_names)
    kl_df_score.to_csv('./leaks/kl_df_score.csv')

    # Get kruskal wallis
    kw_result = kruskal_wallis_distance(score_distributions, leak_names)
    print(f"Kruskal value test for distributions:\nH stat: {kw_result['H_statistic']:.4f}")
    print(f"P value: {kw_result['p_value']:.4e}")
    print(f"Groups: {', '.join(kw_result['groups'])}")

    # Get the colors for the plots
    colors_leaks, colors_categories = get_colors(leak_types)

    # Unused. Useful to plot by length scores and masks
    if LENGTH:
        # Get kl matrix for mask and length
        kl_dfs_mask = compute_kl_matrix_dfs(mask_dataframes, leak_names, 'mask')
        # Get kl matrix for score and length, format np to eliminate np.float values
        kl_dfs_score_length = compute_kl_matrix_dfs(score_length_dataframes, leak_names, 'length')
        # Plot matrices
        plot_by_length(leak_names, kl_dfs_mask, kl_dfs_score_length, figures_folder = 'figures/')
        # Plot and save the score by length distribution
        plot_scores_by_length(score_length_dataframes, leak_names, colors_leaks).savefig(FIGURES_FOLDER + 'bars/scores_length_distribution.png')


    # Plot and save the score distributions
    plot_distributions(score_distributions, leak_names, colors_leaks, colors_categories=colors_categories, categories=leak_categories).savefig(FIGURES_FOLDER + 'bars/scores_distribution.png')
    # Plot and save the kl score matrix
    plot_matrix(kl_df_score.values, leak_names, 'coolwarm', vmin=0, vmax=1).savefig(FIGURES_FOLDER + 'klmatrices/scores_kl_matrix.png')
    # Get box and whiskers plot for values in the score kl matrix
    boxwhiskers_from_kl_matrix(kl_df_score).savefig(FIGURES_FOLDER + 'boxwhiskers/score_boxwhiskers_klmatrix.png')
    # Get scatter from values
    random_scatterplot_klmatrices([kl_df_score], ['All values']).savefig(FIGURES_FOLDER + 'scatter/score_scatter_klmatrix.png')

    kl_matrices_categories = []
    # Get stats for each category
    for category in leak_categories:
        # Get the leaks in the category
        leaks_in_category = [name for name, type in leak_types if type == category]
        # Get a dataframe with the kl values of the category
        category_df = kl_df_score[leaks_in_category].loc[leaks_in_category]
        # Add matrix to category matrices
        kl_matrices_categories.append(category_df)
        # Plot category matrix
        plot_matrix(category_df.values, leaks_in_category, 'coolwarm', vmin=0, vmax=1).savefig(f'{FIGURES_FOLDER}/klmatrices/c_{category}_scores_kl_matrix.png')

    # Plot box and whiskers for each category
    boxwhiskers_from_kl_matrices(kl_matrices_categories, leak_categories, colors_categories).savefig(FIGURES_FOLDER + 'boxwhiskers/categories_boxwhiskers_klmatrices.png')
    # Scatterplot without box whiskers
    random_scatterplot_klmatrices(kl_matrices_categories, leak_categories, colors_categories).savefig(FIGURES_FOLDER + 'scatter/categories_scatter_klmatrices.png')
    plt.close()

    plot_by_year(score_distributions, leak_names, colors_leaks, dates_list).savefig(FIGURES_FOLDER + 'years/years_combined.png')
    plt.close()

    plot_by_year_average(score_distributions, leak_names, colors_leaks, dates_list).savefig(FIGURES_FOLDER + 'years/years_combined_averaged.png')


if __name__ == '__main__':
    get_distribution_comparison()
    
    

    

            

