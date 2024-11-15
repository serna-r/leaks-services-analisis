import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
from packages.retrieve_stats import get_leak_types, get_count_and_probabilities
from packages.plots import plot_regression


CSV_FILE_RISK = 'services\services_risk_dimensions_cluster.csv'
FIGURES_FOLDER = 'figures\leakregression'

def regression(data):
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
    print(f"Slope: {slope}")
    print(f"Intercept: {intercept}")
    print(f"R^2: {r_squared}")

    plot_regression(X,Y,model,slope,intercept).savefig(f"{FIGURES_FOLDER}\\regresion.png")


def leakregression(leaks_file='leak_types.txt'):
    """Function to get the data from the leaks and create a regression between password strength in the leaks and their risk value"""
    # Get leaks
    leak_types = get_leak_types(leaks_file)
    leak_names = [x for x,y in leak_types]
    leak_names.sort()

    # Get risk for leaks, drop cluster column
    services_risks = pd.read_csv(CSV_FILE_RISK)
    services_risks = services_risks.drop('Cluster', axis=1)

    # Get the services corresponding to the leaks
    services_risks = services_risks.loc[services_risks['Website'].str.lower().isin([x.lower() for x in leak_names])]
    services_risks.reset_index(inplace=True)
    services_risks = services_risks.drop('index', axis=1)

    print("Leaks risk:\n", services_risks)

    # Calculate mean score for each leak
    leaks_score_risk = {}
    for leak in leak_names:
        # Calculate leak mean by multiplying probabilities
        scores = [0,1,2,3,4]
        leak_count_prob = get_count_and_probabilities(leak)
        mean = sum([a*b for a,b in zip(leak_count_prob[1],scores)])

        # Get risk
        risk = services_risks.loc[services_risks['Website'].str.lower() == leak.lower(), 'Risk sum'].values
        
        # If risk found append to dict
        if risk:
            risk = int(risk[0])
            # Append mean and risk to dict
            leaks_score_risk[leak] = mean, risk

    regression(leaks_score_risk)

    return