# Standard Libraries
import os
import numpy as np

# Sklearn Modules
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
from sklearn.linear_model import Ridge, Lasso, ElasticNet

# Plotting
import seaborn as sns
import matplotlib.pyplot as plt

def mean_squared_error(
    y : np.ndarray,
    y_pred : np.ndarray
) -> float:
    '''
    '''
    mse = np.mean((y - y_pred)**2)
    return mse

def ridge_regression(
    X : np.ndarray,
    y : np.ndarray
) -> tuple[object, np.ndarray, dict]:
    '''
    '''
    ridge = Ridge()

    param_dist = {
        'alpha': uniform(0, 10),
        'fit_intercept': [True, False],
        'normalize': [True, False]
    }

    rand_search_ridge = RandomizedSearchCV(
        estimator = ridge,
        param_distributions = param_dist,
        n_iter = 100,
        cv = 5,
        random_state = 42
    )
    rand_search_ridge.fit(X, y)

    best_estimator = rand_search_ridge.best_estimator_
    best_estimator_coefficients = rand_search_ridge.best_estimator_.coef_
    best_parameters = rand_search_ridge.best_estimator_.best_params_

    return (best_estimator, best_estimator_coefficients, best_parameters)

def lasso_regression(
    X : np.ndarray,
    y : np.ndarray
) -> tuple[object, np.ndarray, dict]:
    '''
    '''
    lasso = lasso()

    param_dist = {
        'alpha': uniform(0, 10),
        'fit_intercept': [True, False],
        'normalize': [True, False]
    }

    rand_search_lasso = RandomizedSearchCV(
        estimator = lasso,
        param_distributions = param_dist,
        n_iter = 100,
        cv = 5,
        random_state = 42
    )
    rand_search_lasso.fit(X, y)

    best_estimator = rand_search_lasso.best_estimator_
    best_estimator_coefficients = rand_search_lasso.best_estimator_.coef_
    best_parameters = rand_search_lasso.best_estimator_.best_params_

    return (best_estimator, best_estimator_coefficients, best_parameters)

if __name__ == '__main__':
    current_path = os.path.abspath(__file__)
    data_file_path = os.path.abspath(os.path.join(current_path, '..', '..', 'data', 'train.air.csv'))
    data_train = np.loadtxt(data_file_path, delimiter = ',', skiprows = 1)

    X_train = data_train[:, 1:].copy()
    y_train = data_train[:, 0].copy()

    data_file_path = os.path.abspath(os.path.join(current_path, '..', '..', 'data', 'test.air.csv'))
    data_test = np.loadtxt(data_file_path, delimiter = ',', skiprows = 1)

    X_test = data_test[:, 1:].copy()
    y_test = data_test[:, 0].copy()

    # Standardize X
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X)

    # Ridge
    ridge_best_estimator, ridge_best_estimator_coefficients, ridge_best_parameters = ridge_regression(X_train, y_train)
    y_pred = ridge_best_estimator.predict(X_test)
    test_mse = mean_squared_error(y_test, y_pred)
    print(fr'''
    The mean squared error for ridge regression on the test set is:
    {test_mse}
    ''')

    # Lasso
    lasso_best_estimator, lasso_best_estimator_coefficients, lasso_best_parameters = lasso_regression(X_train, y_train)
    y_pred = lasso_best_estimator.predict(X_test)
    test_mse = mean_squared_error(y_test, y_pred)
    print(fr'''
    The mean squared error for lasso regression on the test set is:
    {test_mse}
    ''')
