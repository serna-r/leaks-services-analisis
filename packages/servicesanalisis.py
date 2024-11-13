import pandas as pd
import numpy as np
from packages.plots import plot_all_services, plot_categories_risks, plot_box_whiskers_servicesrisk, plot_radar_risk_dimensions, plot_service_risk_boxplots
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder

# Ignore warning
from scipy.cluster.hierarchy import ClusterWarning
from pandas.errors import SettingWithCopyWarning
from warnings import simplefilter
simplefilter("ignore", ClusterWarning)
simplefilter("ignore", SettingWithCopyWarning)

def get_services_info(file):
    # Get the data
    services = pd.read_excel(file)
    
    # Fill empty cells with 0s
    # Specify the columns by their index positions (7 to 29, inclusive)
    target_columns = services.columns[7:30]  # Python slicing is exclusive at the end, so we use 30

    # Convert all target columns to numeric, coercing errors to NaN if they are non-numeric
    services[target_columns] = services[target_columns].apply(pd.to_numeric, errors='coerce')

    # Now fill NaN values in those columns with 0
    services[target_columns] = services[target_columns].fillna(0)

    return  services

def get_sumservices(services):
    """This function recieves the original data frame and returns the agruppation by services"""
    # Choose relevant columns
    numeric_columns = services.select_dtypes(include=['float64', 'int64']).columns
    sumservices = services[numeric_columns].copy()
    # Drop min length if it is in services
    if 'min length' in sumservices.columns.to_list(): sumservices = sumservices.drop(['min length'], axis=1)

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

def get_data_risk():
    # Get the data
    services_risk = pd.read_excel('./services/risk_dimensions.xlsx', sheet_name='privacy values clean', header=0)
    return services_risk


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

    return manual_silhouette_score

def cluster_services(df):
    # Select only numeric binary columns
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    X = df[numeric_columns].copy()

    # Calculate the pairwise Hamming distance between samples (this is a condensed matrix)
    hamming_dist = pdist(X, metric='hamming')

    # Convert condensed distance matrix to a square matrix for clustering
    hamming_dist_square = squareform(hamming_dist)

    # Variables to track the best Silhouette score and number of clusters
    best_n_clusters = [0,0,0]
    best_score = [-1,-1,-1]

    for n_clusters in range(2, 12):
        # Use precomputed distance (square matrix)
        agg_cluster = AgglomerativeClustering(n_clusters=n_clusters, linkage='complete')
        
        # Fit the model and get the labels
        labels = agg_cluster.fit_predict(hamming_dist_square)  # Pass the square matrix
        
        # Calculate Silhouette score using Hamming distance
        score = silhouette_score(X, labels, metric='hamming')
        
        # Track the best score and number of clusters
        if score > best_score[0]:

            # Assign best scores
            best_score[2] = best_score[1]
            best_score[1] = best_score[0]
            best_score[0] = score

            best_n_clusters[2] = best_n_clusters[1]
            best_n_clusters[1] = best_n_clusters[0]
            best_n_clusters[0] = n_clusters
        # Test for second best score
        elif score > best_score[1]:
            # Assign best scores
            best_score[2] = best_score[1]
            best_score[1] = score

            best_n_clusters[2] = best_n_clusters[1]
            best_n_clusters[1] = n_clusters
        # Test for third best score
        elif score > best_score[2]:
            # Assign best scores
            best_score[2] = score

            best_n_clusters[2] = n_clusters


    # Print the best Silhouette Score and number of clusters
    print(f'Best numbers of clusters: {best_n_clusters}, with Silhouette Scores: {["%.3f" % elem for elem in list(map(float, best_score))]}')

    # Re-fit the clustering with the best number of clusters
    agg_cluster = AgglomerativeClustering(n_clusters=best_n_clusters[0], linkage='complete')
    labels = agg_cluster.fit_predict(hamming_dist_square)

    # Add the cluster labels to the original DataFrame
    df['Cluster'] = labels
    
    return best_n_clusters, best_score


def service_analisis(file):
    # Get the data
    services = get_services_info(file)
    services_only_data = services.iloc[: ,12:30]
    print(services_only_data.dtypes)
    services_only_data[['Type', 'Website']] = services[['Type', 'Website']].copy()

    print(f"\nServices columns: {services.columns.to_list()} \n\nServices only data columns: {services_only_data.columns.to_list()}\n")

    # Get the data collected by category
    # Choose relevant columns
    sumservices = get_sumservices(services)
    sumservices_only_data = get_sumservices(services_only_data)
    # Plot distributions with all data
    plot_all_services(sumservices).savefig('./figures/services/services_data.png')
    # Plot distributions with only personal data
    plot_all_services(sumservices_only_data).savefig('./figures/services/services_personal_data.png')

    # Clusters
    # Manual clusters
    # Evaluate silohuette for categories
    manual_silhouette_score = manual_cluster_evaluation(services)
    # Evaluate silohuette for data
    manual_silhouette_score_data = manual_cluster_evaluation(services_only_data)
    print(f'Manual Silhouette Score (based on Type classification): {manual_silhouette_score:.3f}')
    print(f'Manual Silhouette Score (based on Type classification) only data: {manual_silhouette_score_data:.3f}\n')
    # Automatic clusters
    # Classify all the services
    print('Services clusters')
    cluster_services(services)
    # Classify services only data
    print('Services only data clusters')
    cluster_services(services_only_data)
    # Empty line for formating
    print('')

    # Data risk
    data_risk = get_data_risk()
    data_risk['Total risk'] = data_risk.sum(axis=1, numeric_only=True)

    # Select the relevant columns from both dataframes
    risk_columns = data_risk.select_dtypes(include=['float64', 'int64']).columns.to_list()
    risk_columns.remove('Total risk')
    data_risks_numeric = data_risk[risk_columns]
    data_risks_numeric_max = data_risks_numeric.max().to_frame().T
    services_only_data_numeric = services_only_data[risk_columns]

    # Perform matrix multiplication for each service
    result_sum = {}
    for idx, service in services_only_data.iterrows():
        # Multiply data_risk_risks rows by the service vector (element-wise multiplication)
        multiplied = data_risks_numeric * services_only_data_numeric.iloc[idx].values
        multiplied_max = data_risks_numeric_max * services_only_data_numeric.iloc[idx].values
        
        # Sum the results for each risk dimension
        risk_sum = multiplied.sum(axis=1)
        risk_max_sum = multiplied_max.sum(axis=1)
        
        # Store the result with the service's name
        result_sum[service['Website']] = risk_sum.values

    # Convert the result dictionary into a DataFrame for better readability
    services_risk_dimensions = pd.DataFrame(result_sum, index=data_risk['Risk dimension']).transpose()
    services_risk_dimensions.index.name = 'Website'
    services_risk_dimensions['Type'] = services_only_data.set_index('Website')['Type']

    # Get total risk
    services_risk_dimensions['Risk sum'] = services_risk_dimensions.sum(axis=1, numeric_only = True)

    print(f'Maximum for each dimension \n{services_risk_dimensions.max().to_frame().T}')

    # Group risks by category and get mean
    categories_group = services_risk_dimensions.groupby('Type')
    categories_risk = categories_group.mean(numeric_only=True)
    # Reset the index to move 'Type' from index to a column
    categories_risk = categories_risk.reset_index()

    # Plots
    # Plot the barplot for categories with the mean
    plot_categories_risks(categories_risk).savefig('./figures/services/bars_categories_risk.png')
    # Boxplot for each type of service with values for dimensions
    plot_service_risk_boxplots(services_risk_dimensions).savefig('./figures/services/boxwhiskers_services_risk_dimensions.png')
    # Boxplot for each type of service for the sum of the dimensions
    plot_box_whiskers_servicesrisk(services_risk_dimensions).savefig('./figures/services/boxwhisker_services_risk.png')
    # Radar plot for types
    plot_radar_risk_dimensions(categories_risk).savefig('./figures/services/radar_services_risk.png')

    # Format df for output
    services_risk_dimensions.reset_index()
    cols = services_risk_dimensions.columns.to_list()
    cols.remove('Type')
    cols.insert(0, 'Type')
    services_risk_dimensions = services_risk_dimensions[cols]
    # Add cluster label
    services_risk_dimensions['Cluster'] = services_only_data.set_index('Website').loc[:, 'Cluster']

    # Save csvs for output
    # Save services with clusters
    services.to_csv('./services/servicescluster.csv')
    # Save services only data with clusters
    services_only_data.to_csv('./services/services_only_data_cluster.csv')
    # Save services risk dimensions
    services_risk_dimensions.to_csv('./services/services_risk_dimensions_cluster.csv')


    
  
if __name__ == '__main__':

    service_analisis('./services/services.xlsx')