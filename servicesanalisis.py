import pandas as pd
import numpy as np
from plots import plot_all_services, plot_categories_risks, plot_box_whiskers_servicesrisk
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder

# Ignore warning
from scipy.cluster.hierarchy import ClusterWarning
from warnings import simplefilter
simplefilter("ignore", ClusterWarning)

def get_services_info(file):
    # Get the data
    services = pd.read_excel(file)
    
    # Fill empty cells with 0s
    for i in range(20):
        services.iloc[:,i+7] = services.iloc[:,i+7].fillna(0)

    return  services

def get_sumservices(services):
    """This function recieves the original data frame and returns the agruppation by services"""
    # Choose relevant columns
    sumservices = services.iloc[:, 7:27]
    sumservices['Type'] = services['Type']
    # Add total column to store total of services in column
    sumservices['Total'] = 1.0
    # Group by service
    sumservices = sumservices.groupby(['Type']).sum()
    # Normalize
    sumservices = sumservices.div(sumservices['Total'], axis=0)
    # Eliminate total column
    sumservices = sumservices.loc[:, sumservices.columns != 'Total']

    return sumservices

def get_services_risk(file):
    # Get the data
    services_risk = pd.read_excel(file, header=0)

    # Return website and risk
    return  services_risk[['Website','Type', 'Risk']]

def manual_cluster_evaluation(services):
    # Get new dataframe
    df = services.copy()
    # Select only numeric binary columns for clustering
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    X = df[numeric_columns]

    # Encode the 'Type' column into numeric labels (e.g., "adult" -> 0, "social" -> 1)
    label_encoder = LabelEncoder()
    df['Type_encoded'] = label_encoder.fit_transform(df['Type'])

    # Calculate the pairwise Hamming distance between samples
    hamming_dist = pdist(X, metric='hamming')

    # Manual evaluation: Calculate silhouette score based on the provided 'Type' labels
    manual_silhouette_score = silhouette_score(X, df['Type_encoded'], metric='hamming')

    print(f'Manual Silhouette Score (based on Type classification): {manual_silhouette_score}')

def cluster_services(df):
    # Select only numeric binary columns
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    X = df[numeric_columns]

    # Calculate the pairwise Hamming distance between samples (this is a condensed matrix)
    hamming_dist = pdist(X, metric='hamming')

    # Convert condensed distance matrix to a square matrix for clustering
    hamming_dist_square = squareform(hamming_dist)

    # Variables to track the best Silhouette score and number of clusters
    best_n_clusters = 0
    best_score = -1

    for n_clusters in range(2, 20):
        # Use precomputed distance (square matrix)
        agg_cluster = AgglomerativeClustering(n_clusters=n_clusters, linkage='complete')
        
        # Fit the model and get the labels
        labels = agg_cluster.fit_predict(hamming_dist_square)  # Pass the square matrix
        
        # Calculate Silhouette score using Hamming distance
        score = silhouette_score(X, labels, metric='hamming')
        
        # Track the best score and number of clusters
        if score > best_score:
            best_score = score
            best_n_clusters = n_clusters

    # Print the best Silhouette Score and number of clusters
    print(f'Best number of clusters: {best_n_clusters}, with Silhouette Score: {best_score}')

    # Re-fit the clustering with the best number of clusters
    agg_cluster = AgglomerativeClustering(n_clusters=best_n_clusters, linkage='complete')
    labels = agg_cluster.fit_predict(hamming_dist_square)

    # Add the cluster labels to the original DataFrame
    df['Cluster'] = labels
    
    return best_n_clusters, best_score


def service_analisis(file):
    # Get the data
    services = get_services_info(file)

    # Get the data collected by category
    # Choose relevant columns
    sumservices = get_sumservices(services)
    # Plot distributions with all data
    plot_all_services(sumservices).savefig('./figures/services/services.png')
    # Plot distributions with only personal data
    plot_all_services(sumservices.iloc[:, 2:]).savefig('./figures/services/services_personal.png')
    # Save csv
    sumservices.to_csv('./services/servicestypesum.csv')

    # Evaluate silohuette for categories
    manual_cluster_evaluation(services)
    # Classify all the services
    cluster_services(services)
    services.to_csv('./services/servicescluster.csv')
    
    # Get the risks by category
    # Get the services with the risks
    services_risk = get_services_risk('./services/risk_dimensions.xlsx')
    # Group risks by category and get mean
    categories_group = services_risk.groupby('Type')
    categories_risk = categories_group.mean(numeric_only=True)
    # Reset the index to move 'Type' from index to a column
    categories_risk = categories_risk.reset_index()
    # Plot
    plot_categories_risks(categories_risk).savefig('./figures/services/categories_risk.png')
    plot_box_whiskers_servicesrisk(services_risk).savefig('./figures/services/services_risk_boxwhiskers.png')
    
  
if __name__ == '__main__':

    service_analisis('./services/services.xlsx')