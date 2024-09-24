from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

import numpy as np
from collections import defaultdict
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from retrieve_stats import get_count_and_probabilities, get_leak_types

verbose = 0

def get_leaks_and_probabilities(leak_types):
    # Lists for returning
    leak_names = []
    leak_probabilities = []
    # Get probabilities for all services
    for leak, category in leak_types:
        # Get leak file name
        leak_file = './leaks/' + leak + '/Stats.txt'
        # Open file and get stats
        with open(leak_file, 'r') as f:
            # Read file content
            data = f.read()
            
            # Get score probabilities
            count_list, probability_dist = get_count_and_probabilities(data)
            # Append to list
            leak_probabilities.append(probability_dist)
            leak_names.append(leak)

    return leak_names, leak_probabilities

# Function to group by category
def group_by_category(leak_list, key_index):
    category_dict = defaultdict(list)
    for item in leak_list:
        category_dict[item[key_index]].append(item[0])
    return category_dict

def get_elbow_kmeans(leaks_file):
    """Function to print the sse for each value of k"""
    # Get leak names
    leak_types = get_leak_types(leaks_file)
    leak_types.sort(key=lambda x: x[1])
    # Get names and probabilities
    leak_names, leak_probabilities = get_leaks_and_probabilities(leak_types)

    number_leaks = len(leak_names)

    kmeans_kwargs = {
        "init": "random",
        "n_init": 10,
        "max_iter": 300,
        "random_state": 42,
    }
   
    # A list holds the SSE values for each k
    sse = []
    for k in range(1, number_leaks):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(leak_probabilities)
        sse.append(kmeans.inertia_)

    plt.style.use("fivethirtyeight")
    plt.plot(range(1, number_leaks), sse)
    plt.xticks(range(1, number_leaks))
    plt.xlabel("Number of Clusters")
    plt.ylabel("SSE")
    return number_leaks, plt

def execute_kmeans(leaks_file, k):
    # Get leak names
    leak_types = get_leak_types(leaks_file)
    leak_types.sort(key=lambda x: x[1])
    # Get names and probabilities
    leak_names, leak_probabilities = get_leaks_and_probabilities(leak_types)

    # Initiate kmeans
    kmeans = KMeans(
        init="random",
        n_clusters=k,
        n_init=10,
        max_iter=300,
        random_state=42
    )

    # Fit the distributions
    kmeans.fit(leak_probabilities)

    if verbose > 0:
        kmeans_labels = list(zip(leak_names, kmeans.labels_.tolist()))
        # Group by category for leak_types
        leak_types_grouped = group_by_category(leak_types, 1)
        # Group by kmeans label for kmeans_labels
        kmeans_grouped = group_by_category(kmeans_labels, 1)

        # Printing the results
        print("Leak Types by Category:")
        for category, names in leak_types_grouped.items():
            print(f"{category}: {', '.join(names)}")

        print("\nLeak Types by KMeans Label:")
        for label, names in kmeans_grouped.items():
            print(f"{label}: {', '.join(names)}")

    sse = kmeans.inertia_
    # Return leakname with category, kmeans label and probabilities
    return sse, list(zip(leak_types, kmeans.labels_.tolist(), leak_probabilities))

def plot_kmeans(data, sse):
    # Sort data by category
    data_sorted = sorted(data, key=lambda x: x[1])

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

if __name__ == '__main__':
    # Leaks file
    leaks_file='leak_types.txt'

    # Get elbow for kmeans
    number_leaks, elbowplot = get_elbow_kmeans(leaks_file)
    # Save elbow figure
    elbowplot.savefig('./figures/kmeans/elbow.png')

    for i in range (1, int(number_leaks/2)):
        # Execute k means
        sse, kmeans_data = execute_kmeans(leaks_file, k=i)

        # Plot kmeans
        plot_kmeans(kmeans_data, sse).savefig(f'./figures/kmeans/kmeans_k{i}.png')

