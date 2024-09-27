from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

import numpy as np
from collections import defaultdict
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from plots import plot_kmeans, plot_5d_scatter
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

def get_centroids(leak_types):
    centroids = []

    return centroids


def get_elbow_kmeans(leak_names, leak_probabilities):
    """Function to print the sse for each value of k"""

    number_leaks = len(leak_names)

    kmeans_kwargs = {
        "init": "random",
        "n_init": 10,
        "max_iter": 300,
        "random_state": 42,
    }
   
    # A list holds the SSE values for each k
    sse = []
    for k in range(2, number_leaks):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(leak_probabilities)
        sse.append(kmeans.inertia_)

    plt.style.use("fivethirtyeight")
    plt.plot(range(2, number_leaks), sse)
    plt.xticks(range(2, number_leaks))
    plt.xlabel("Number of Clusters")
    plt.ylabel("SSE")
    return number_leaks, plt

def execute_kmeans(leak_types, leak_names, leak_probabilities, k):

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

    # Get sse
    sse = kmeans.inertia_
    # Get centroids
    centroids = kmeans.cluster_centers_
    # Return leakname with category, kmeans label and probabilities
    return sse, list(zip(leak_types, kmeans.labels_.tolist(), leak_probabilities)), centroids

def get_kmeans(leak_names, leak_types,  leak_probabilities):
    # Get elbow for kmeans
    number_leaks, elbowplot = get_elbow_kmeans(leak_names, leak_probabilities)
    # Save elbow figure
    elbowplot.savefig('./figures/kmeans/elbow.png')
    plt.close()

    for i in range (2, int(number_leaks/2)):
        # Execute k means
        sse, kmeans_data, centroids = execute_kmeans(leak_types, leak_names, leak_probabilities, k=i)

        categories = [data[1] for data in kmeans_data]
        plot_5d_scatter(leak_names, leak_probabilities, categories, centroids).savefig(f"./figures/kmeans/services_5d_scatter_{i}.png")
        plt.close()

        # Plot kmeans
        plot_kmeans(kmeans_data, sse).savefig(f'./figures/kmeans/kmeans_k{i}.png')
        plt.close()

def manual_cluster(leak_types):
    # Get unique categories and map each to an integer
    unique_categories = sorted(set([leak[1] for leak in leak_types]))
    category_to_label = {category: idx for idx, category in enumerate(unique_categories)}
    
    # Replace the category in leak_types with its corresponding integer label
    numerical_labels = [category_to_label[category] for _, category in leak_types]
    
    return numerical_labels, category_to_label


def cluster_evaluation(numerical_labels, leak_probabilities):
    # Group the leak probabilities by their numerical label
    clusters = defaultdict(list)
    
    for i, label in enumerate(numerical_labels):
        clusters[label].append(leak_probabilities[i])
    
    # Calculate the centroid for each cluster
    centroids = {label: np.mean(probs, axis=0) for label, probs in clusters.items()}
    
    # Calculate the overall mean of the dataset
    all_data_points = [prob for probs in leak_probabilities for prob in probs]
    overall_mean = np.mean(all_data_points, axis=0)
    
    # Sum of Squared Within (SSW)
    ssw = 0
    for label, probs in clusters.items():
        centroid = centroids[label]
        for prob in probs:
            ssw += np.sum((prob - centroid) ** 2)
    
    # Sum of Squared Between (SSB)
    ssb = 0
    for label, probs in clusters.items():
        centroid = centroids[label]
        n_j = len(probs)
        ssb += n_j * np.sum((centroid - overall_mean) ** 2)

    # Create centroids as array to return
    centroids_list = np.asarray([centroids[i].tolist() for i in range(len(centroids))])
    
    return ssw, ssb, centroids_list

def clustering(leaks_file, kmeans=False):
    # Get leak names
    leak_types = get_leak_types(leaks_file)
    leak_types.sort(key=lambda x: x[1])
    # Get names and probabilities
    leak_names, leak_probabilities = get_leaks_and_probabilities(leak_types)
    # Plot 4d scatter of the leaks
    plot_5d_scatter(leak_names, leak_probabilities).savefig("./figures/scatter/services_5d_scatter.png")
    plt.close()

    # If needed apply kmeans
    if kmeans: get_kmeans(leak_names, leak_types, leak_probabilities)

    # Get numerical labels for each category
    numerical_labels, category_to_label = manual_cluster(leak_types)
    print(f"Labels for each category:\n{category_to_label}")
    # Call manual clustering
    ssw, ssb, centroids = cluster_evaluation(numerical_labels, leak_probabilities)
    # Print measures
    print(f"Manual Measures SSW: {ssw} , SSB: {ssb}")

    # Plot manual cluster
    plot_5d_scatter(leak_names, leak_probabilities, numerical_labels).savefig("./figures/scatter/services_5d_manualcluster_scatter.png")
    plt.close()

    # Execute kmeans 5
    sse, cluster_list, centroidsK = execute_kmeans(leak_types, leak_names, leak_probabilities, 5)
    ssw2, ssb2, centroids2 = cluster_evaluation([item[1] for item in cluster_list], [item[2] for item in cluster_list])
    print(f"Kmeans Measures SSW: {ssw2} , SSB: {ssb2}")

if __name__ == '__main__':
    # Leaks file
    leaks_file='leak_types.txt'

    clustering(leaks_file)
    

