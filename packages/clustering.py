import numpy as np
from collections import defaultdict
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.metrics import davies_bouldin_score
from packages.plots import plot_kmeans, plot_5d_scatter
from packages.retrieve_stats import get_count_and_probabilities, get_leak_types

verbose = 0
OUTPUTFOLDER = './clusters/'
FINALKMEANS = 6
FIGURES_FOLDER = './figures/cluster/'


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
            count_list, probability_dist = get_count_and_probabilities(leak)
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
    plt.figure(figsize=(16, 8))   # make figure larger
    plt.plot(range(2, number_leaks), sse)
    plt.xticks(range(2, number_leaks))
    plt.xlabel("Number of Clusters")
    plt.ylabel("SSE")
    return number_leaks, plt

def get_silhouette_kmeans(leak_names, leak_probabilities):
    """
    Function to plot the silhouette score for different values of k in k-means clustering.
    
    Args:
        leak_names (list): List of leak names (used to determine the number of data points).
        leak_probabilities (list of list): The feature vectors for clustering.
    
    Returns:
        plt: The silhouette plot for different numbers of clusters.
    """
    number_leaks = len(leak_names)

    kmeans_kwargs = {
        "init": "random",
        "n_init": 10,
        "max_iter": 300,
        "random_state": 42,
    }
    
    # A list holds the silhouette score values for each k
    silhouette_scores = []
    for k in range(2, number_leaks):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(leak_probabilities)
        score = silhouette_score(leak_probabilities, kmeans.labels_)
        silhouette_scores.append(score)
    
    # Plotting the silhouette scores for each k
    plt.style.use("fivethirtyeight")
    plt.plot(range(2, number_leaks), silhouette_scores)
    plt.xticks(range(2, number_leaks))
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Score for KMeans Clustering")

    # Return plot
    return plt

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
    # Save elbow figure for sse
    elbowplot.savefig(FIGURES_FOLDER + 'kmeans/elbow.png')
    plt.close()
    # Save  figure for silhouette
    get_silhouette_kmeans(leak_names, leak_probabilities).savefig(FIGURES_FOLDER + 'kmeans/silhouette_kmeans.png')

    plt.close()

    for i in range (2, int(number_leaks/2)):
        # Execute k means
        sse, kmeans_data, centroids = execute_kmeans(leak_types, leak_names, leak_probabilities, k=i)

        categories = [data[1] for data in kmeans_data]
        plot_5d_scatter(leak_names, leak_probabilities, categories, centroids).savefig(f"{FIGURES_FOLDER}kmeans/services_5d_scatter_{i}.png")
        plt.close()

        # Plot kmeans
        plot_kmeans(kmeans_data, sse).savefig(f'{FIGURES_FOLDER}kmeans/kmeans_k{i}.png')
        plt.close()

def manual_cluster_labels(leak_types):
    # Get unique categories and map each to an integer
    unique_categories = sorted(set([leak[1] for leak in leak_types]))
    category_to_label = {category: idx for idx, category in enumerate(unique_categories)}
    
    # Replace the category in leak_types with its corresponding integer label
    numerical_labels = [category_to_label[category] for _, category in leak_types]
    
    return numerical_labels, category_to_label

class ClusterEvaluation:
    def __init__(self, leak_names, numerical_labels, leak_probabilities):
        self.leak_names = leak_names
        self.numerical_labels = numerical_labels
        self.leak_probabilities = leak_probabilities
        self.k = len(set(numerical_labels))  # Number of clusters
        self.n = len(leak_names)  # Total number of data points
        self.d = len(leak_probabilities[0])  # Dimensionality (number of features per point)

    def evaluate(self):
        # Calculate SSW, SSB, and centroids
        self.ssw, self.ssb, self.centroids = self.calculate_ssw_ssb()

        # Calculate all indices
        self.ball_hall = self.ball_hall_index()
        self.calinski_harabasz = self.calinski_harabasz_index()
        self.hartigan = self.hartigan_index()
        self.davies_bouldin = self.davies_bouldin_index()
        self.silhouette, self.silhouette_values = self.silhouette_index_and_values()

        return {
            'Ball-Hall': self.ball_hall,
            'Calinski-Harabasz': self.calinski_harabasz,
            'Hartigan': self.hartigan,
            'Davies-Bouldin': self.davies_bouldin,
            'Silhouette': self.silhouette
        }

    def calculate_ssw_ssb(self):
        # Group leak probabilities by their cluster label
        clusters = defaultdict(list)

        for i, label in enumerate(self.numerical_labels):
            clusters[label].append(self.leak_probabilities[i])

        # Calculate the centroid for each cluster
        centroids = {label: np.mean(probs, axis=0) for label, probs in clusters.items()}

        # Calculate the overall mean of the dataset
        all_data_points = np.vstack(self.leak_probabilities)
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
            n_j = len(probs)  # Number of points in the cluster
            ssb += n_j * np.sum((centroid - overall_mean) ** 2)

        centroids_array = np.array([centroids[i] for i in range(self.k)])
        return ssw, ssb, centroids_array

    # Ball-Hall index (1965)
    def ball_hall_index(self):
        return self.ssw / self.k

    # Calinski-Harabasz index (1974)
    def calinski_harabasz_index(self):
        return (self.ssb / (self.k - 1)) / (self.ssw / (self.n - self.k))

    # Hartigan index (1975)
    def hartigan_index(self):
        return np.log(self.ssb / self.ssw)

    # Davies-Bouldin index
    def davies_bouldin_index(self):
        # Call sklearn library
        db_index = davies_bouldin_score(self.leak_probabilities, self.numerical_labels)
        # Return value
        return db_index

    # Silhouette index
    def silhouette_index_and_values(self):
        silhouette_index = silhouette_score(self.leak_probabilities, self.numerical_labels)
        silhouette_values = silhouette_samples(self.leak_probabilities, self.numerical_labels)
        return silhouette_index, silhouette_values 
    
    def plot_silhouette(self, vmin=None, vmax=None):
        # Convert numerical_labels to NumPy array for easier handling
        cluster_labels = np.array(self.numerical_labels)

        # Create a color map with distinct colors for each cluster
        colors = cm.get_cmap("tab10", self.k)  # Use a predefined color map with distinct colors

        # Create a figure and axis for the plot
        fig, ax = plt.subplots(figsize=(10, 7))

        # Create grid
        plt.grid()

        # Sort silhouette values for better visualization
        y_positions = np.arange(len(self.silhouette_values))  # One bar for each service

        # Plot a bar for each silhouette score
        for i in range(self.n):
            cluster = cluster_labels[i]
            ax.barh(y_positions[i], self.silhouette_values[i], color=colors(cluster), edgecolor='black')

        # Set labels and title
        ax.set_yticks(y_positions)
        ax.set_yticklabels(self.leak_names)  # Use leak names as labels for each bar
        ax.set_xlabel("Silhouette Score")
        ax.set_ylabel("Services (Leaks)")
        ax.set_title("Silhouette Plot for Each Service by Cluster")

        # Set the x-axis limits based on provided vmin and vmax
        if vmin is not None and vmax is not None:
            ax.set_xlim(vmin, vmax)

        # Display the plot
        plt.tight_layout()
        plt.grid()

        # Return plot
        return plt

    # Method to pretty print all the calculated indices
    def __str__(self):
        return (f"Cluster Evaluation Indices:\n"
                f"SSW: {self.ssw}, SSB: {self.ssb}\n"
                f"Ball-Hall Index: {self.ball_hall:.4f}\n"
                f"Calinski-Harabasz Index: {self.calinski_harabasz:.4f}\n"
                f"Hartigan Index: {self.hartigan:.4f}\n"
                f"Davies-Bouldin Index: {self.davies_bouldin:.4f}\n"
                f"Silhouette Index: {self.silhouette:.4f}")

    

def clustering(leaks_file, kmeans=False):
    # Get leak names
    leak_types = get_leak_types(leaks_file)
    leak_types.sort(key=lambda x: x[1])
    # Get names and probabilities
    leak_names, leak_probabilities = get_leaks_and_probabilities(leak_types)
    # Plot 4d scatter of the leaks
    plot_5d_scatter(leak_names, leak_probabilities).savefig(FIGURES_FOLDER + "scatter/services_5d_scatter.png")
    plt.close()

    # If needed apply kmeans to get the best k
    if kmeans: get_kmeans(leak_names, leak_types, leak_probabilities)

    # Get numerical labels for each category
    numerical_labels, category_to_label = manual_cluster_labels(leak_types)
    print(f"Labels for each category:\n{category_to_label}\n")
    # Call manual clustering
    manualCluster = ClusterEvaluation(leak_names, numerical_labels, leak_probabilities)
    manualCluster.evaluate()
    manualCluster.plot_silhouette(vmin=-0.7,vmax=0.7).savefig(FIGURES_FOLDER + "bars/silohuetteManual.png")
    # Print measures
    print(f"Manual Measures")
    print(manualCluster, "\n")
    # Save indexes in file
    with open(OUTPUTFOLDER + 'manualcluster_evaluation.txt', 'w') as f: f.write(str(manualCluster))
    # Plot manual cluster
    plot_5d_scatter(leak_names, leak_probabilities, numerical_labels).savefig(FIGURES_FOLDER + "scatter/services_5d_manualcluster_scatter.png")
    plt.close()

    # Execute kmeans FINALKMEANS
    sse, cluster_list, centroidsKmeans = execute_kmeans(leak_types, leak_names, leak_probabilities, FINALKMEANS)
    kmeansCluster = ClusterEvaluation(leak_names, [item[1] for item in cluster_list], leak_probabilities)
    kmeansCluster.evaluate()
    kmeansCluster.plot_silhouette(vmin=-0.7,vmax=0.7).savefig(f"{FIGURES_FOLDER}bars/silohuetteKmeans{FINALKMEANS}.png")
    # Plot bars cluster sorted
    sorted_kmeans = sorted(list(zip(leak_types, kmeansCluster.numerical_labels, leak_probabilities)), key=lambda x: x[1])
    plot_kmeans(sorted_kmeans, kmeansCluster.ssw, default_sort=False).savefig(f"{FIGURES_FOLDER}kmeans/kmeans_k{FINALKMEANS}_sorted.png")
    # Print stats for kmeans
    print(f"Kmeans {FINALKMEANS} Measures")
    print(kmeansCluster, "\n")
    # Plot kmeans scatter
    plot_5d_scatter(leak_names, leak_probabilities, kmeansCluster.numerical_labels, centroidsKmeans).savefig(f"{FIGURES_FOLDER}scatter/services_5d_kmeanscluster_scatter.png")

    # Save indexes in file
    with open(OUTPUTFOLDER + f'kmeans{FINALKMEANS}cluster_evaluation.txt', 'w') as f: f.write(str(kmeansCluster))


if __name__ == '__main__':
    # Leaks file
    leaks_file='leak_types.txt'

    clustering(leaks_file)
    

