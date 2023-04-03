# Standard Libraries
import os
import numpy as np

# Disable warning messages
import warnings
warnings.filterwarnings("ignore")

# Sklearn Modules
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression, lasso_path

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
        'alpha': np.concatenate((np.arange(0.01, 1, 0.2), np.arange(0, 10, 0.5))),
        'fit_intercept': [True, False]
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
    best_parameters = rand_search_ridge.best_params_

    return (best_estimator, best_estimator_coefficients, best_parameters)

def lasso_regression(
    X : np.ndarray,
    y : np.ndarray
) -> tuple[object, np.ndarray, dict]:
    '''
    '''
    lasso = Lasso()

    param_dist = {
        'alpha': np.concatenate((np.arange(0.01, 1, 0.2), np.arange(0, 10, 0.5))),
        'fit_intercept': [True, False]
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
    best_parameters = rand_search_lasso.best_params_

    return (best_estimator, best_estimator_coefficients, best_parameters)

class AdaptiveLasso(BaseEstimator, RegressorMixin):
    def __init__(
        self
    ) -> None:
        '''
        '''
        self.ols_betas = None
        self.gamma = None
        self.best_estimator = None

    def adaptive_lasso_regression(
        self,
        X : np.ndarray,
        y : np.ndarray,
        gamma : float
    ) -> tuple[object, np.ndarray, dict]:
        '''
        '''
        self.gamma = gamma

        self.ols_betas = LinearRegression(fit_intercept = False).fit(X, y).coef_
        w_ols = self.ols_betas ** (-self.gamma)

        X_ols = X.copy() / w_ols

        lambdas, lasso_betas, _ = lasso_path(X_ols, y)
        
        lasso = Lasso()

        param_dist = {
            'alpha': lambdas,
            'fit_intercept': [True, False]
        }

        rand_search_adaptive_lasso = RandomizedSearchCV(
            estimator = lasso,
            param_distributions = param_dist,
            n_iter = 100,
            cv = 5,
            random_state = 42
        )
        rand_search_adaptive_lasso.fit(X_ols, y)

        self.best_estimator = rand_search_adaptive_lasso.best_estimator_
        best_estimator_coefficients = self.best_estimator.coef_
        best_parameters = rand_search_adaptive_lasso.best_params_

        return (self.best_estimator, best_estimator_coefficients, best_parameters)

    def predict(
        self,
        X : np.ndarray
    ) -> np.ndarray:
        '''
        '''
        w_ols = self.ols_betas ** (-self.gamma)
        X_ols = X.copy() / w_ols
        return  self.best_estimator.predict(X_ols)

def elastic_net_regression(
    X : np.ndarray,
    y : np.ndarray
) -> tuple[object, np.ndarray, dict]:
    '''
    '''
    elastic_net = ElasticNet()

    param_dist = {
        'alpha': np.concatenate((np.arange(0.01, 1, 0.2), np.arange(0, 10, 0.5))),
        'l1_ratio': np.arange(0, 1.1, 0.1),
        'fit_intercept': [True, False]
    }

    rand_search_elastic_net = RandomizedSearchCV(
        estimator = elastic_net,
        param_distributions = param_dist,
        n_iter = 100,
        cv = 5,
        random_state = 42
    )
    rand_search_elastic_net.fit(X, y)

    best_estimator = rand_search_elastic_net.best_estimator_
    best_estimator_coefficients = rand_search_elastic_net.best_estimator_.coef_
    best_parameters = rand_search_elastic_net.best_params_

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
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Ridge
    ridge_best_estimator, ridge_best_estimator_coefficients, ridge_best_parameters = ridge_regression(X_train, y_train)
    y_pred = ridge_best_estimator.predict(X_test)
    test_mse = mean_squared_error(y_test, y_pred)
    print(fr'''
    The mean squared error for ridge regression on the test set is: {round(test_mse, 5)}
    ''')

    # Lasso
    lasso_best_estimator, lasso_best_estimator_coefficients, lasso_best_parameters = lasso_regression(X_train, y_train)
    y_pred = lasso_best_estimator.predict(X_test)
    test_mse = mean_squared_error(y_test, y_pred)
    print(fr'''
    The mean squared error for lasso regression on the test set is: {round(test_mse, 5)}
    ''')

    # Adaptive Lasso
    gamma = 2
    a_lasso = AdaptiveLasso()
    adaptive_lasso_best_estimator, adaptive_lasso_best_estimator_coefficients, adaptive_lasso_best_parameters = a_lasso.adaptive_lasso_regression(X_train, y_train, gamma)
    y_pred = a_lasso.predict(X_test)
    test_mse = mean_squared_error(y_test, y_pred)
    print(fr'''
    The mean squared error for adaptive lasso regression on the test set is: {round(test_mse, 5)}
    ''')

    # Elastic Net
    elastic_net_best_estimator, elastic_net_best_estimator_coefficients, elastic_net_best_parameters = elastic_net_regression(X_train, y_train)
    y_pred = elastic_net_best_estimator.predict(X_test)
    test_mse = mean_squared_error(y_test, y_pred)
    print(fr'''
    The mean squared error for elastic net regression on the test set is: {round(test_mse, 5)}
    ''')

