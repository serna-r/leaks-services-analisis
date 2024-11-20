import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
from packages.retrieve_stats import get_leak_types, get_count_and_probabilities
from packages.plots import plot_regression
from packages.servicesanalisis import get_services_info


CSV_FILE_RISK = 'services\services_risk_dimensions_cluster.csv'
FIGURES_FOLDER = 'figures\leakregression'

def regression(data, label):
    """Fuction to create a regression from a dict and print results"""

    # Split data into X and Y
    X = np.array([val[0] for val in data.values()]).reshape(-1, 1)  # Mean
    Y = np.array([val[1] for val in data.values()])  # Score

    # Fit a linear regression model
    model = LinearRegression()
    model.fit(X, Y)

    # Regression line parameters
    slope = model.coef_[0]
    intercept = model.intercept_
    r_squared = model.score(X, Y)

    # Print results
    print(f"Regresion: {label}")
    print(f"Slope: {slope}")
    print(f"Intercept: {intercept}")
    print(f"R^2: {r_squared}\n")

    return plot_regression(X,Y,model,slope,intercept, label)


def leakregression(leaks_file='leak_types.txt'):
    """Function to get the data from the leaks and create a regression between password strength in the leaks and their risk value"""
    # Get leaks
    leak_types = get_leak_types(leaks_file)
    leak_names = [x for x,y in leak_types]
    leak_names.sort()

    # Get number of data colected by each leak
    service_data ='./services/services.xlsx'
    services = get_services_info(service_data)
    services_only_data = services.iloc[: ,12:30]
    services_only_data['sum'] = services_only_data.sum(axis=1, numeric_only=True)
    services_only_data['Website'] = services.loc[:, 'Website']
    services_count = services_only_data.loc[:, ['Website', 'sum']]

    # Get the services corresponding to the leaks
    services_count = services_count.loc[services_count['Website'].str.lower().isin([x.lower() for x in leak_names])]
    services_count.reset_index(inplace=True)
    services_count = services_count.drop('index', axis=1)

    print("Leaks count:\n", services_count)

    # Get risk for leaks, drop cluster column
    services_risks = pd.read_csv(CSV_FILE_RISK)
    services_risks = services_risks.drop('Cluster', axis=1)

    # Get the services corresponding to the leaks
    services_risks = services_risks.loc[services_risks['Website'].str.lower().isin([x.lower() for x in leak_names])]
    services_risks.reset_index(inplace=True)
    services_risks = services_risks.drop('index', axis=1)

    print("\nLeaks risk:\n", services_risks)

    # Calculate mean score for each leak
    leaks_score_risk = {}
    leaks_score_count = {}
    for leak in leak_names:
        # Calculate leak mean by multiplying probabilities
        scores = [0,1,2,3,4]
        leak_count_prob = get_count_and_probabilities(leak)
        mean = sum([a*b for a,b in zip(leak_count_prob[1],scores)])

        # Get risk
        risk = services_risks.loc[services_risks['Website'].str.lower() == leak.lower(), 'Risk sum'].values

        # Get number of data collected
        count = services_count.loc[services_count['Website'].str.lower() == leak.lower(), 'sum'].values
        
        # If risk found append to dict
        if risk:
            risk = int(risk[0])
            # Append mean and risk to dict
            leaks_score_risk[leak] = mean, risk
            leaks_score_count[leak] = mean, count

    # Regression with scores
    risk_regression = regression(leaks_score_risk, 'Risk')
    risk_regression.savefig(f"{FIGURES_FOLDER}\\risk_regresion.png")
    risk_regression.close() 

    # Regression with count
    count_regression = regression(leaks_score_count, 'Data collected count')
    count_regression.savefig(f"{FIGURES_FOLDER}\\count_regresion.png")
    count_regression.close()
    
    return