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
    # Plot distributions with all data
    plot_all_services(sumservices).savefig('./figures/services/services.png')
    # Plot distributions with only personal data
    plot_all_services(sumservices.iloc[:, 2:]).savefig('./figures/services/services_personal.png')
    # Save csv
    sumservices.to_csv('./services/servicestypesum.csv')

if __name__ == '__main__':

    service_analisis('./services/services.xlsx')