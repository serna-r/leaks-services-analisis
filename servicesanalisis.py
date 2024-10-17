import pandas as pd
import numpy as np
from plots import plot_all_services, plot_categories_risks, plot_box_whiskers_servicesrisk

def get_services_info(file):
    # Get the data
    services = pd.read_excel(file)

    # Fill empty cells with 0s
    for i in range(15):
        services.iloc[:,i+7] = services.iloc[:,i+7].fillna(0)

    return  services

def get_services_risk(file):
    # Get the data
    services_risk = pd.read_excel(file, header=0)

    # Return website and risk
    return  services_risk[['Website','Type', 'Risk']]

def service_analisis(file):
    # Get the data
    services = get_services_info(file)
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