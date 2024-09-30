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
    # Get mean of non 0 values
    non0mean = distribution[distribution!= 0].mean()
    # Add titles and labels
    plt.title(f"Categories and Values: {title}, Non 0 mean: {non0mean:.4f}")
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
    # Eliminate total column
    sumservices = sumservices.loc[:, sumservices.columns != 'Total']
    # Plot distributions
    for i in range(len(sumservices.index)):
        category = sumservices.iloc[i].name
        plot_services_sum(sumservices.iloc[i, :].values, sumservices.columns, category).savefig(f'./figures/services/{category}.png')
        plt.close()
    # Save csv
    sumservices.to_csv('./services/servicestypesum.csv')

if __name__ == '__main__':

    service_analisis('./services/services.xlsx')