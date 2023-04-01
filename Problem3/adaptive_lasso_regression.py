import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression, lasso_path

class AdaptiveLasso(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        ols_betas : np.ndarray,
        gamma : float,
        best_estimator : object
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