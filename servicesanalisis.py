import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_data(file):
    # Get the data
    services = pd.read_excel(file)

    # Fill empty cells with 0s
    for i in range(15):
        services.iloc[:,i+7] = services.iloc[:,i+7].fillna(0)

    return  services

def plot_services_sum(distribution, names, title):
    # Create a horizontal bar chart using pyplot
    plt.barh(names, distribution, edgecolor='black')

    # Add titles and labels
    plt.title("Histogram of Categories and Values")
    plt.xlabel("Values")
    plt.ylabel("Categories")

    # Show the plot
    plt.tight_layout()

    return plt

def service_analisis(file):
    # Get the data
    services = get_data(file)
    # Group by service
    sumservices = services.iloc[:, 7:21]
    sumservices['Type'] = services['Type']
    sumservices['Total'] = 1.0
    sumservices = sumservices.groupby(['Type']).sum()
    # Normalize and get important columns
    sumservices.iloc[:, 0:] = sumservices.iloc[:, 0:].div(sumservices['Total'], axis=0)
    # Plot distributions
    for i in range(len(sumservices.index)):
        plot_services_sum(sumservices.iloc[i, :].values, sumservices.columns, 'Test').savefig(f'./figures/services/{i}.png')
        plt.close()
    # Save csv
    sumservices.to_csv('./services/servicestypesum.csv')

if __name__ == '__main__':

    service_analisis('./services/services.xlsx')