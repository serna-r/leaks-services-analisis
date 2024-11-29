import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from packages.retrieve_stats import get_leak_types, get_count_and_probabilities
from packages.plots import plot_regression
from packages.servicesanalisis import get_services_info

LEAKS_FILE = 'leak_types.txt'
CSV_FILE_RISK = 'services\services_risk_dimensions_cluster.csv'
CSV_SERVICE_DATA ='.\services\services.xlsx'
FIGURES_FOLDER = 'figures\leakregression'
VERBOSE = 0

def xy_regression(data, label, xlabel='x', ylabel='y'):
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

    return plot_regression(X,Y,model,slope,intercept, label, xlabel, ylabel)

def multivariate_regression(df, label, xlabel, ylabel):
    """Calculate multivariate regression with the data collected vs strength, without train-test split."""
    # Drop non necessary columns
    df = df.drop(columns=['sum', 'Website'])

    # Split into features (X) and target (y)
    X = df.drop(columns=['strength'])  # All Boolean predictors
    y = df['strength']

    # Train the regression model on the full dataset
    model = LinearRegression()
    model.fit(X, y)

    # Evaluate the model on the full dataset
    r2 = model.score(X, y)

    print('\nMultivariate regression data collected vs strength')
    print("R-squared:", r2)

    # Get coefficients to understand feature importance
    coefficients = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': model.coef_
    }).sort_values(by='Coefficient', ascending=False)

    print(coefficients)

    # Re-run with only important coefficients

    # Define a threshold for important features
    importance_threshold = 0.4

    # Filter important features based on the absolute value of coefficients
    important_features = coefficients[coefficients['Coefficient'].abs() > importance_threshold]['Feature'].tolist()

    print("\nImportant Features:", important_features)

    # Filter the DataFrame to keep only important features
    X_important = X[important_features]

    # Train the regression model with important features on the full dataset
    model_imp = LinearRegression()
    model_imp.fit(X_important, y)

    # Evaluate the reduced model
    r2_imp = model_imp.score(X_important, y)

    print("\nR-squared:", r2_imp)

    # Get new coefficients
    coefficients_imp = pd.DataFrame({
        'Feature': X_important.columns,
        'Coefficient': model_imp.coef_
    }).sort_values(by='Coefficient', ascending=False)

    print(coefficients_imp)

    return



def leakregression(leaks_file=LEAKS_FILE):
    """Function to get the data from the leaks and create a regression between password strength in the leaks and their risk value"""
    # Get leaks
    leak_types = get_leak_types(leaks_file)
    leak_names = [x for x,y in leak_types]
    leak_names.sort()

    # Get the services data stored
    services = get_services_info(CSV_SERVICE_DATA)
    services['Website'] = services['Website'].str.lower()

    # Get the services corresponding to leaks
    leaked_services = services.loc[services['Website'].isin([x for x in leak_names])]
    leaked_services.reset_index(inplace=True)
    leaked_services_only_data = leaked_services.iloc[: ,13:30]

    if VERBOSE > 1: print("\nleaked services only data:\n", leaked_services_only_data)

    # Get number of data colected by each leak
    leaked_services_only_data['sum'] = leaked_services_only_data.sum(axis=1, numeric_only=True)
    leaked_services_only_data['Website'] = leaked_services.loc[:, 'Website']
    services_count = leaked_services_only_data.loc[:, ['Website', 'sum']]

    # Get the services corresponding to the leaks
    services_count.reset_index(inplace=True)
    services_count = services_count.drop('index', axis=1)

    if VERBOSE > 0: print("Leaks count:\n", services_count)

    # Get risk for leaks, drop cluster column
    services_risks = pd.read_csv(CSV_FILE_RISK)
    services_risks = services_risks.drop(['Only data cluster', 'NIST Compliance Score'], axis=1)

    # Get the services corresponding to the leaks
    services_risks['Website'] = services_risks['Website'].str.lower().str.strip()
    services_risks = services_risks.loc[services_risks['Website'].isin([x for x in leak_names])]
    services_risks.reset_index(inplace=True)
    services_risks = services_risks.drop('index', axis=1)

    if VERBOSE > 0: print("\nLeaks risk:\n", services_risks)

    # Calculate mean score for each leak and store it in dict with risk and count
    leaks_score_risk = {}
    leaks_score_count = {}
    for leak in leak_names:
        # Calculate leak mean by multiplying probabilities
        scores = [0,1,2,3,4]
        leak_count_prob = get_count_and_probabilities(leak)
        mean = sum([a*b for a,b in zip(leak_count_prob[1],scores)])

        # Get risk
        risk = services_risks.loc[services_risks['Website'] == leak, 'Risk sum'].values

        # Get number of data collected
        count = services_count.loc[services_count['Website'] == leak, 'sum'].values
        
        # If risk found append to dict
        if risk:
            risk = int(risk[0])
            # Append mean and risk to dict
            leaks_score_risk[leak] = mean, risk
            leaks_score_count[leak] = mean, count

    # Add the strength to the data frame
    leaked_services_only_data['strength'] = leaked_services_only_data['Website'].map(lambda x: leaks_score_risk[x][0] if x in leaks_score_risk else None)

    # Regression with scores
    risk_regression = xy_regression(leaks_score_risk, 'Risk vs stregnth', xlabel='Password strength', ylabel='Service risk')
    risk_regression.savefig(f"{FIGURES_FOLDER}\\risk_regresion.png")
    risk_regression.close() 

    # Regression with count
    count_regression = xy_regression(leaks_score_count, 'Data collected count vs strenght', xlabel='Password strength', ylabel='Service data collected count')
    count_regression.savefig(f"{FIGURES_FOLDER}\\count_regresion.png")
    count_regression.close()

    # Multivariate regression type of data collected vs strength
    datacollected_regression = multivariate_regression(leaked_services_only_data, 'Data collected vs strength', xlabel='Data collected', ylabel='Strength')
    # datacollected_regression.savefig(f"{FIGURES_FOLDER}\\datacollected_regression.png")
    # datacollected_regression.close()
    
    return