import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator, RegressorMixin

class AdaptiveLasso(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        alpha : float = 1.0,
        gamma : float = 1.0,
        max_iter : int = 1000,
        tol : float = 1e-6,
        verbose : bool = False
    ) -> None:
        '''
        '''
        self.alpha = alpha
        self.gamma = gamma
        self.max_iter = max_iter
        self.tol = tol
        self.beta = None
        self.verbose = verbose

    def loss_function(
        self,
        y : np.ndarray,
        y_pred : np.ndarray
    ):
        '''
        '''
        return p.mean((y - y_pred)**2)

    def loss_gradient(
        self,
        y : np.ndarray,
        y_pred : np.ndarray,
        X : np.ndarray
    ):
        '''
        '''
        return -2 * np.mean((y - y_pred)[:, np.newaxis] * X, axis = 0)

    def adaptive_weights(
        self,
        beta : np.ndarray
    ) -> np.ndarray:
        '''
        '''
        return np.abs(beta) ** (self.gamma - 1)

    def coordinate_gradient_descent(
        self,
        X : np.ndarray,
        y : np.ndarray
    ) -> None:
        '''
        '''
        n_samples, n_features = X.shape
        self.beta = np.zeros(n_features)
        y_pred = np.dot(X, self.beta)
        
        for iteration in range(self.max_iter):
            beta_old = self.beta.copy()
            
            for j in range(n_features):
                y_pred = y_pred - X[:, j] * self.beta[j]
                gradient = self.loss_gradient(y, y_pred, X[:, j])
                z_j = np.sum(X[:, j] ** 2)
                weight = self.adaptive_weights(self.beta[j])
                self.beta[j] = np.sign(gradient) * max(0, np.abs(gradient) - self.alpha * weight) / z_j
                y_pred = y_pred + X[:, j] * self.beta[j]
            
            if self.verbose:
                print(np.linalg.norm(self.beta - beta_old))
            
            if np.linalg.norm(self.beta - beta_old) < self.tol:
                break

    def fit(
        self,
        X : np.ndarray,
        y : np.ndarray
    ):
        '''
        '''
        self.coordinate_gradient_descent(X, y)
        return self

    def predict(
        self,
        X : np.ndarray
    ) -> np.ndarray:
        '''
        '''
        return np.dot(X, self.beta)