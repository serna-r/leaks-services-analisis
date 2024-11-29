import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import to_rgba
import matplotlib.patches as mpatches
from datetime import datetime
from math import pi

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
    for category in dict(sorted(category_count.items(), key=lambda d: d[0])):
        # Get colormap
        colormap = base_colors[category]
        # Select color in the middle
        color = colormap(0.70)
        
        # Append the color
        colors_categories.append(color)

    return colors_leaks, colors_categories
            

def plot_distributions(distributions, names, colors=None, colors_categories = None, categories = None, year=None):
    num_distributions = len(distributions)
    num_categories = len(distributions[0])  # Each distribution has 5 score categories

    # Positions for the bars
    x = np.arange(num_categories)  # Category positions on the x-axis
    bar_width = 0.8 / num_distributions  # Width of each bar

    fig, ax = plt.subplots(figsize=(12, 7))

    for i, distribution in enumerate(distributions):
        plt.bar(
            x + i * bar_width,  # Shift each service's bars to the right
            distribution,
            bar_width,
            label=names[i].capitalize(),  # Use the service's name as the label
            color=colors[i] if colors else None  # Use provided colors, if any
        )

    plt.xlabel('Scores (zxcvbn)')
    plt.ylabel('Probability')

    if year is not None:
        plt.title(f'Probability Distributions for {year}')
    else:
        plt.title(f'Probability Distributions')

    # Set x-ticks to be centered with proper labels (Score 0, Score 1, ...)
    plt.xticks(x + bar_width * (num_distributions - 1) / 2, [f'Score {i}' for i in range(num_categories)])

    # Legend for categories (Second Legend)
    if categories is not None and colors_categories is not None:
        # Format strings
        categories = [cat.capitalize() for cat in categories]
        categories = [cat.replace('Digitaltool', 'Digital Tool') for cat in categories]
        # Create legend handles with color patches
        category_handles = [mpatches.Patch(color=color, label=category) for color, category in zip(colors_categories, categories)]
        categories_legend = ax.legend(handles=category_handles, title="Categories", loc='upper right')
        ax.add_artist(categories_legend)
    # Move the legend above the title and split it into multiple rows if it's too long
    ncols = min(4, num_distributions)  # Set max columns to 4
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.55), ncol=ncols)


    # Adjust the layout to add more space above the plot for the legend and title
    plt.subplots_adjust(top=0.85)

    plt.tight_layout()
    return plt

def plot_by_year_average(score_distributions, leak_names, colors_leaks, dates_list):
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

    # Sort years, with "Unknown" last
    known_years = sorted([year for year in data_by_year.keys() if year != "Unknown"])
    if "Unknown" in data_by_year:
        years = known_years + ["Unknown"]
    else:
        years = known_years

    num_categories = len(score_distributions[0])  # Assuming all distributions have the same number of categories

    # Plotting
    plt.figure(figsize=(12, 7))

    # Positions for the bars
    x = np.arange(num_categories)  # Category positions on the x-axis
    bar_width = 0.8 / len(years)  # Width of each bar per year

    # Plot bars ordered by year
    for i, year in enumerate(years):
        data = data_by_year[year]
        combined_distribution = np.mean(data["distributions"], axis=0)  # Average distribution for the year
        plt.bar(
            x + i * bar_width, 
            combined_distribution, 
            bar_width, 
            label=str(year), 
            color=data["colors"][0]  # Use the first color for the year (assuming consistency in color for each year)
        )

    plt.xlabel('Scores (zxcvbn)')
    plt.ylabel('Probability')

    plt.title('Probability Distributions average by Year')

    # Set x-ticks to be centered with proper labels (Score 0, Score 1, ...)
    plt.xticks(x + bar_width * (len(years) - 1) / 2, [f'Score {i}' for i in range(num_categories)])

    # Move the legend to the top, with years in rows if there are too many
    ncols = min(4, len(years))  # Set max columns to 4
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=ncols)

    # Adjust layout to make space for legend
    plt.subplots_adjust(top=0.85)
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
            data_by_year[year] = {"distributions": [], "leak_names": []}
        data_by_year[year]["distributions"].append(score_distributions[i])
        data_by_year[year]["leak_names"].append(leak_names[i])

    # Sort years, with "Unknown" last
    known_years = sorted([year for year in data_by_year.keys() if year != "Unknown"])
    if "Unknown" in data_by_year:
        years = known_years + ["Unknown"]
    else:
        years = known_years

    num_categories = len(score_distributions[0])  # Assuming all distributions have the same number of categories

    # Generate distinct colors
    colors = plt.cm.viridis(np.linspace(0, 1, len(years)))  # Using the 'viridis' colormap for better visibility

    # Plotting
    plt.figure(figsize=(12, 7))

    # Positions for the bars
    x = np.arange(num_categories)  # Category positions on the x-axis
    bar_width = 0.8 / sum(len(data_by_year[year]["distributions"]) for year in years)  # Bar width

    total_distributions = 0  # Counter to correctly position bars

    # Plot bars for each year and its corresponding distributions
    for idx, year in enumerate(years):
        data = data_by_year[year]
        n_distributions = len(data["distributions"])

        for i in range(n_distributions):
            distribution = data["distributions"][i]
            plt.bar(
                x + total_distributions * bar_width, 
                distribution, 
                bar_width, 
                label=data["leak_names"][i] if i == 0 else "",  # Label each leak in the legend, only once per year
                color=colors[idx],  # Use the generated color
                alpha=0.9
            )
            total_distributions += 1

    plt.xlabel('Scores (zxcvbn)')
    plt.ylabel('Probability')

    plt.title('Probability Distributions Ordered and Colored by Year')

    # Set x-ticks to be centered with proper labels (Score 0, Score 1, ...)
    plt.xticks(x + bar_width * (total_distributions - 1) / 2, [f'Score {i}' for i in range(num_categories)])

    # Create a custom legend for the years using the colors from the generated list
    plt.legend([f'Year {year}' for year in years], loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=min(4, len(years)))

    # Adjust layout to make space for the legend
    plt.subplots_adjust(top=0.85)
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
    ax.set_xlabel(f'Q0.25 (green) {q025:.2f}, Q0.15 (yellow) {q015:.2f}, Q0.10 (red) {q010:.2f}')

    # Add top margin
    plt.subplots_adjust(top=0.8)

    # Annotate each cell with the numerical value and add points if lower than quantile
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f'{data[i, j]:.2f}', ha='center', va='center', color='white', fontsize=10)
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

    # Create space in digital tool
    labels = [label.replace('digitaltool', 'digital tool') for label in labels]

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

def plot_kmeans(data, sse, default_sort = True):
    # Sort data by category
    if default_sort: data_sorted = sorted(data, key=lambda x: x[0][0])
    else: data_sorted = data

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

        if centroids is not None:
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

def plot_categories_risks(categories_risks):
    plt.figure(figsize=(10,6))
    cmap = plt.get_cmap('tab20')

    # Order values
    categories_risks = categories_risks.sort_values('Risk sum', ascending=False)

    X = categories_risks.iloc[:, 0]
    labels = categories_risks['Risk sum']
    plt.bar(X, labels, color=cmap.colors)

    # Adding labels and title
    plt.xlabel('Category')
    plt.ylabel('Value')
    plt.title('Category Value Distribution')
    plt.xticks(rotation=45, ha='right')

    # Display the plot
    plt.tight_layout()
    
    return plt

def plot_box_whiskers_servicesrisk(services_risk):
    # Get colors
    cmap = plt.get_cmap('tab20')
    colors = cmap.colors
    
    # Get unique service types ordered by mean
    categories_group = services_risk.groupby('Type').mean(numeric_only=True).reset_index().sort_values('Risk sum', ascending=False)
    labels = categories_group['Type']
    # Clean nan
    labels = [x for x in labels if str(x) != 'nan']
    
    # Collect values grouped by service type
    values = [services_risk[services_risk['Type'] == service]['Risk sum'].values for service in labels]

    # Create a boxplot
    plt.figure(figsize=(10, 8))
    plt.boxplot(values, showmeans=True)
    plt.title('Box-and-Whiskers Plot of Risk by Service Type')
    plt.ylabel('Risk')
    plt.xticks(np.array([i for i in range(len(labels))]) + 1, labels, rotation=45, ha='right', fontsize=8)
    plt.grid(True)

    # Add scatter plot of individual points, color-coded
    for i in range(len(labels)):
        y = values[i]
        x = np.random.normal(1 + i, 0.04, size=len(y))
        plt.plot(x, y, '.', alpha=0.8, color=colors[i] if colors else None)

    return plt

def plot_radar_risk_dimensions(categories_risk):
    # Extract the numeric values to plot
    plotable = categories_risk.select_dtypes(include=[np.number])[['Physical', 'Social', 'Resources', 'Psychological', 'Prosecution', 'Career', 'Freedom']]
    categories = plotable.columns.to_list()
    N = len(categories)

    # Create angles for the radar chart
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]  # Complete the loop to close the chart

    # Split into two DataFrames: high and low values
    overall_mean = plotable.mean().mean()
    top_values = plotable[plotable.mean(axis=1) > overall_mean]
    bottom_values = plotable[plotable.mean(axis=1) <= overall_mean]

    # Define a common scale range for both charts
    max_value = plotable.max().max()
    min_value = plotable.min().min()

    # Create the figure with two subplots
    fig, axs = plt.subplots(1, 2, figsize=(16, 8), subplot_kw=dict(polar=True))
    fig.suptitle("Radar Plots: High and Low Risk Values", fontsize=16)

    # Helper function to plot radar with data bubbles
    def add_radar_chart(ax, data, title, unique_labels):
        colors = plt.cm.plasma(np.linspace(0, 1, len(data)))  # Generate distinct colors for each line

        for index, (row, color) in enumerate(zip(data.iterrows(), colors)):
            row = row[1]
            values = row.tolist() + row.tolist()[:1]
            label = unique_labels[index]  # Use unique labels for each subset
            
            # Plot the radar line with markers for each data point
            ax.plot(angles, values, 'o-', ms=4, mec="w", linewidth=1.5, color=color, clip_on=False, zorder=1, label=label)

            # Add bubbles and larger labels for visibility
            for angle, value in zip(angles, values):
                ax.plot([angle], [value], 'o', ms=10, mec="w", color=color, zorder=2)  # Draw bubble
                ax.annotate(label[0], xy=(angle, value), color="w", size=6, ha="center", va="center")  # Larger font size for initials

        ax.set_ylim(min_value, max_value + 1)  # Keep a common scale for both plots, increase max
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_title(title, fontsize=14)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)

    # Unique labels for high and low values to avoid duplication in legends
    top_labels = categories_risk['Type'][top_values.index].unique()
    bottom_labels = categories_risk['Type'][bottom_values.index].unique()

    # Radar chart for high values
    add_radar_chart(axs[0], top_values, 'High Values', top_labels)

    # Radar chart for low values
    add_radar_chart(axs[1], bottom_values, 'Low Values', bottom_labels)

    return plt


def plot_service_risk_boxplots(df):
    # Filter only the risk dimension columns and type
    risk_columns = ['Physical', 'Social', 'Resources', 'Psychological', 'Prosecution', 'Career', 'Freedom']
    
    # Calculate rows and columns for the subplots
    num_categories = len(risk_columns)
    nrows = (num_categories + 1) // 2  # Two columns per row
    fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(12, nrows * 5), sharey=True)
    axes = axes.flatten()  # Flatten to easily iterate with a single index
    
    # Loop through each risk dimension to create a boxplot for each, grouped by 'Type'
    for i, risk in enumerate(risk_columns):
        # Plot boxplot for each risk dimension, grouped by Type
        df.boxplot(column=risk, by='Type', ax=axes[i])
        
        # Set axis labels and title
        axes[i].set_title(f'{risk} Risk')
        axes[i].set_ylabel('Risk Value')
        
        # Set x-axis labels at the top and rotate them
        axes[i].xaxis.set_label_position('top')
        axes[i].xaxis.tick_top()
        axes[i].set_xlabel('Service Type')
        for label in axes[i].get_xticklabels():
            label.set_rotation(45)
            label.set_ha('right')  # Align labels to the right for better readability
    
    # Adjust layout for readability
    plt.suptitle('Risk Values by Service Type and Risk Dimension', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    return plt


def plot_regression(X, Y, model, slope, intercept, label, xlabel, ylabel):
    """Function to plot regression line and data points"""
    # Ensure slope and intercept are scalars
    slope_scalar = slope.item() if hasattr(slope, 'item') else slope
    intercept_scalar = intercept.item() if hasattr(intercept, 'item') else intercept
    
    # Plot data points only for two variable regression
    if not isinstance(X, pd.DataFrame):
        plt.scatter(X, Y, color='blue', label='Data Points')
    
    # Plot regression line
    plt.plot(X, model.predict(X), color='red', label=f'Regression Line: y={slope_scalar:.2f}x + {intercept_scalar:.2f}')
    
    # Add labels and legend
    plt.title(f"Regression Plot: {label}")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    
    # Return plot
    return plt


def boxplot_nist_compliance(df):
    # Group data by type for the plot
    categories_group = df.groupby('Type').mean(numeric_only=True).reset_index().sort_values('NIST Compliance Score', ascending=False)
    labels = categories_group['Type']

    # Clean NaNs
    labels = [x for x in labels if str(x) != 'nan']

    # Collect compliance scores grouped by type
    values = [df[df['Type'] == service_type]['NIST Compliance Score'].values for service_type in labels]

    # Get colors
    cmap = plt.get_cmap('tab20')
    colors = cmap.colors

    # Create a boxplot
    plt.figure(figsize=(10, 8))
    plt.boxplot(values, showmeans=True)
    plt.title('Box-and-Whiskers Plot of NIST Compliance by Service Type')
    plt.ylabel('Compliance Score')
    plt.xticks(np.array([i for i in range(len(labels))]) + 1, labels, rotation=45, ha='right', fontsize=8)
    plt.grid(True)

    # Add scatter plot of individual points, color-coded
    for i in range(len(labels)):
        y = values[i]
        x = np.random.normal(1 + i, 0.04, size=len(y))  # Scatter around the boxplot positions
        plt.plot(x, y, '.', alpha=0.8, color=colors[i % len(colors)] if colors else None)

    # Show the plot
    plt.tight_layout()

    return plt