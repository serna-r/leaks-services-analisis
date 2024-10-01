import pandas as pd
import numpy as np
from plots import plot_all_services

def get_data(file):
    # Get the data
    services = pd.read_excel(file)

    # Fill empty cells with 0s
    for i in range(15):
        services.iloc[:,i+7] = services.iloc[:,i+7].fillna(0)

    return  services

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
    plot_all_services(sumservices).savefig('./figures/services/services.png')
    # Save csv
    sumservices.to_csv('./services/servicestypesum.csv')

if __name__ == '__main__':

    service_analisis('./services/services.xlsx')