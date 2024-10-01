import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import to_rgba
from datetime import datetime

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
    
    # Colors for leaks
    colors_leaks = []
        
    # For each line in the file give a color for a leak
    for leak in leak_types:
        category = leak[-1]  # The category is the second item of the dict
        
        # Increment the count for this category to vary the brightness
        category_count[category] += 1
        
        # Get a color from the colormap with different brightness
        colormap = base_colors[category]
        color = colormap(category_count[category] / len([l for l in leak_types if category in l]))  # Normalized brightness level
        
        # Append the color
        colors_leaks.append(to_rgba(color))

    # colors_categories
    colors_categories = []

    # Get a color for each category
    for category in category_count:
        # Get colormap
        colormap = base_colors[category]
        # Select color in the middle
        color = colormap(0.99)
        
        # Append the color
        colors_categories.append(color)
    
    
    return colors_leaks, colors_categories
            

def plot_distributions(distributions, names, colors=None, year=None):
    num_distributions = len(distributions)
    num_categories = len(distributions[0])  # Each distribution has 5 score categories
    
    # Positions for the bars
    x = np.arange(num_categories)  # Category positions on the x-axis
    bar_width = 0.8 / num_distributions  # Width of each bar

    plt.figure(figsize=(12, 6))
    
    for i, distribution in enumerate(distributions):
        plt.bar(
            x + i * bar_width,  # Shift each service's bars to the right
            distribution, 
            bar_width, 
            label=names[i],  # Use the service's name as the label
            color=colors[i] if colors else None  # Use provided colors, if any
        )
    
    plt.xlabel('Scores (zxcvbn)')
    plt.ylabel('Probability')

    if year != None : plt.title(f'Probability Distributions for {year}')
    else: plt.title(f'Probability Distributions')
    
    # Set x-ticks to be centered with proper labels (Score 0, Score 1, ...)
    plt.xticks(x + bar_width * (num_distributions - 1) / 2, [f'Score {i}' for i in range(num_categories)])
    
    plt.legend(loc='best')
    plt.tight_layout()
    return plt

def plot_by_year(score_distributions, leak_names, colors_leaks, dates_list):
    # Organize data by year
    data_by_year = {}
    for i, date in enumerate(dates_list):
        try:
            year = datetime.strptime(date, "%d/%m/%Y").year
        except ValueError:
            year = "Unknown"
        
        if year not in data_by_year:
            data_by_year[year] = {"distributions": [], "leak_names": [], "colors": []}
        data_by_year[year]["distributions"].append(score_distributions[i])
        data_by_year[year]["leak_names"].append(leak_names[i])
        data_by_year[year]["colors"].append(colors_leaks[i])

    # Separate and sort years, handling "Unknown" separately
    known_years = sorted([year for year in data_by_year.keys() if year != "Unknown"])
    if "Unknown" in data_by_year:
        years = known_years + ["Unknown"]
    else:
        years = known_years
    
    n_years = len(years)
    
    # Create subplots with two columns
    nrows = (n_years + 1) // 2  # Number of rows based on the number of years
    fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(12, nrows * 5), sharex=True, sharey=True)

    # Flatten axes for easy iteration
    axes = axes.flatten()

    # Plot bars side by side for each year
    for idx, year in enumerate(years):
        ax = axes[idx]
        data = data_by_year[year]
        
        n_distributions = len(data["distributions"])
        width = 0.8 / n_distributions  # Make sure the bars fit within the x-axis
        
        for i, dist in enumerate(data["distributions"]):
            x = np.arange(len(dist))  # Position on x-axis
            ax.bar(x + i * width, dist, width=width, color=data["colors"][i], alpha=0.7, label=data["leak_names"][i])
        
        ax.set_title(f"Year: {year}")
        ax.legend()
    
    # Hide any unused subplots if the number of years is odd
    for i in range(len(years), len(axes)):
        fig.delaxes(axes[i])
    
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
        plot_matrix(item[1].values, leak_names, 'coolwarm').savefig(figures_folder + f'klmatrices/mask_length_{item[0]}_kl_matrix.png')

    # For each score and length plot the matrix
    for item in kl_dfs_score_length:
        plot_matrix(item[1].values, leak_names, 'coolwarm').savefig(figures_folder + f'klmatrices/score_length_{item[0]}_kl_matrix.png')


def boxwhiskers_from_kl_matrix(kl_matrix):
    # Get values without the main diagonal (as it is always o)
    non_diag_values = kl_matrix.to_numpy()[~np.eye(kl_matrix.shape[0], dtype=bool)]

    # Create boxplot
    plt.figure(figsize=(8, 6))
    plt.boxplot(non_diag_values, showmeans='True')
    plt.title('Box and Whiskers Plot (Ignoring Diagonal)')
    plt.xticks([1],['All kl values'])
    plt.ylabel('Values')
    plt.grid(True)

    # Plot points for visualization
    x = np.random.normal(1, 0.04, size=len(non_diag_values))
    plt.plot(x, non_diag_values, 'r.', alpha=0.2)

    # Show mean and quantiles
    q010 = np.quantile(non_diag_values, .10)
    q015 = np.quantile(non_diag_values, .15)
    q1 = np.quantile(non_diag_values, .25)
    q2 = np.quantile(non_diag_values, .50)
    q3 = np.quantile(non_diag_values, .75)
    plt.text(1, -0.25, f'mean: {np.mean(non_diag_values)}, Q1 {q1:.5f}, Q2 {q2:.5f}, Q3 {q3:.5f} \n Q0.10 {q010:.5f}, Q0.15 {q015:.5f}', horizontalalignment='center')
    
    return plt

def boxwhiskers_from_kl_matrices(kl_matrices, labels, colors=None):
    # Get values without the main diagonal (as it is always o)
    values = []
    for matrix in kl_matrices:
        non_diag_values = matrix.to_numpy()[~np.eye(matrix.shape[0], dtype=bool)]
        values.append(non_diag_values)

    # Create boxplot
    plt.figure(figsize=(8, 6))
    plt.boxplot(values, showmeans='True')
    plt.title('Box and Whiskers Plot (Ignoring Diagonal)')
    plt.ylabel('Values')
    plt.xticks(np.array([i for i in range(len(labels))]) + 1 , labels, rotation=45, ha='right', fontsize=8)
    plt.grid(True)

    # Plot points for visualization
    for i in range(len(labels)):
        y = values[i]
        x = np.random.normal(1+i, 0.04, size=len(y))
        plt.plot(x, y, '.', alpha=0.5, color=colors[i] if colors else None)
    
    return plt

def random_scatterplot_klmatrices(kl_matrices, labels, colors = None):
    # Get values without the main diagonal (as it is always o)
    values = []
    for matrix in kl_matrices:
        non_diag_values = matrix.to_numpy()[~np.eye(matrix.shape[0], dtype=bool)]
        values.append(non_diag_values)

    # Create boxplot
    plt.figure(figsize=(8, 6))
    plt.title('Categories Scatterplot (Ignoring Diagonal)')
    plt.ylabel('Values')
    plt.xticks(np.array([i for i in range(len(labels))]) + 1 , labels, rotation=45, ha='right', fontsize=8)
    plt.grid(True)

    # Plot points for visualization
    for i in range(len(labels)):
        y = values[i]
        x = np.random.normal(1+i, 0.04, size=len(y))
        plt.plot(x, y, '.', alpha=0.5, color=colors[i] if colors else None)
    
    return plt

def plot_kmeans(data, sse):
    # Sort data by category
    data_sorted = sorted(data, key=lambda x: x[0][0])

    # Extract sorted labels, categories, and values
    labels = [f"{entry[0][0]} ({entry[0][1]})" for entry in data_sorted]  # Individual (Category)
    categories_list = [entry[1] for entry in data_sorted]  # Category numbers
    values = [entry[2] for entry in data_sorted]  # The 5 values for each individual

    # Convert the list of values into a numpy array for easier plotting
    values = np.array(values)

    # Dynamically generate colors for the categories
    unique_categories = sorted(set(categories_list))  # Find unique categories and sort them
    colors = plt.get_cmap('tab10', len(unique_categories))  # Use a colormap with the number of unique categories
    category_colors = {category: colors(i) for i, category in enumerate(unique_categories)}  # Assign a color to each category

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 7))

    # Create a stacked bar for each category
    bottom = np.zeros(len(values))  # To start stacking from 0 for each individual

    for i in range(values.shape[1]):  # Iterate over the 5 values per individual
        color_list = [category_colors[cat] for cat in categories_list]  # Color depends on category
        ax.bar(labels, values[:, i], bottom=bottom, color=color_list, edgecolor='white')
        bottom += values[:, i]  # Update bottom for the next stacked value

    # Add labels and title
    ax.set_xlabel(f'Services, SSE = {sse}')
    ax.set_ylabel('Distribution')
    ax.set_title('Distribution of Categories Across Services (Ordered by K means label)')

    # Show the plot
    plt.xticks(rotation=90)
    plt.tight_layout()
    return plt

def plot_5d_scatter(labels, data, categories=None, centroids=None):
    
    # Convert to numpy array for easier indexing
    data = np.array(data)

    # Extract the first three dimensions for x, y, z
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]

    # Use the 4th dimension to represent color
    color = data[:, 3]

    # Use the 5th dimension to represent size (scaled for better visualization)
    size = data[:, 4] * 1000  # Adjust scaling factor if needed

    fig = plt.figure(figsize=(7, 7), dpi=300)
    ax = fig.add_subplot(111, projection='3d')

    # Set color for categories if necesarie
    if categories:
        unique_categories = sorted(set(categories))
        num_categories = len(set(categories))
        # Get distinct colors for each category (using a colormap)
        colors = plt.get_cmap('Set1', num_categories)
        category_colors = {category: colors(i) for i, category in enumerate(unique_categories)}   # Assign a color to each category

    # Create the scatter plot, with size and color representing the 4th and 5th dimensions
    if categories:
        vmax = max(color)
        vmin = min(color)
        for i in range(len(x)):
            ax.scatter(x[i], y[i], z[i], c=color[i], s=size[i], cmap='viridis', alpha=0.7,
                    edgecolors=category_colors[categories[i]], linewidth=1.2, vmax=vmax, vmin=vmin)

        # Add a color bar to show the scale of the 4th dimension (color)
        scatter = ax.scatter([], [], [], c=[], cmap='viridis')  # Dummy scatter for the colorbar
        colorbar = plt.colorbar(scatter, location='left', fraction=0.03, pad=0.03)

        if centroids != None:
            for i, centroid in enumerate(centroids):
                ax.scatter(centroid[0],centroid[1], centroid[2], c=[centroid[3]], s=(centroid[4]* 1000), cmap='viridis', alpha=0.7,
                        linewidth=1.2, marker='+', vmax=vmax, vmin=vmin)
    else:
        scatter = ax.scatter(x, y, z, c=color, s=size, cmap='viridis', alpha=0.7)
        # Add a color bar to show the scale of the 4th dimension (color)
        colorbar = plt.colorbar(scatter, location='left', fraction=0.03, pad=0.03)



    # Set axis labels
    ax.set_xlabel('Score 0')
    ax.set_ylabel('Score 1')
    ax.set_zlabel('Score 2')
    colorbar.set_label('Score 3')

    # Add labels next to each point
    for i, label in enumerate(labels):
        ax.text(x[i], y[i], z[i], label, fontsize=3, color='black')

    return plt

def plot_all_services(distribution_df):
    # Sort categories (rows) alphabetically
    sorted_df = distribution_df.sort_index()

    # Number of categories (rows)
    num_categories = len(sorted_df.index)
    
    # Create subplots with two columns
    nrows = (num_categories + 1) // 2  # Two columns
    fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(12, nrows * 5), sharex=True, sharey=True)

    # Flatten axes for easy iteration
    axes = axes.flatten()

    # Iterate through each category and plot
    for i, (category, row) in enumerate(sorted_df.iterrows()):
        ax = axes[i]
        distribution = row.values
        names = row.index
        
        # Plot the horizontal bar chart
        ax.barh(names, distribution, edgecolor='black')
        
        # Get mean of non-zero values
        non0mean = distribution[distribution != 0].mean() if np.any(distribution != 0) else 0
        
        # Set title and labels
        ax.set_title(f"{category} (Non-0 mean: {non0mean:.4f})")
        ax.set_xlabel("Values")
        ax.set_ylabel("Categories")

    # Hide any unused subplots if the number of categories is odd
    for i in range(len(sorted_df), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    
    return plt
    