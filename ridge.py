import numpy as np
from sklearn.model_selection import KFold

class RidgeRegression:
    def __init__(self, alpha=1.0, n_splits=5, random_state=None):
        self.alpha = alpha
        self.n_splits = n_splits
        self.random_state = random_state
        self.weights = None
        self.mse = None
        self.rmse = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        X = np.column_stack([np.ones(n_samples), X])
        I = np.identity(n_features + 1)
        I[0][0] = 0
        
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        best_score = float("-inf")
        best_weights = None

        mse = []

        for train_index, val_index in kf.split(X):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]

            XTX = np.dot(X_train.T, X_train)
            weights = np.linalg.inv(XTX + self.alpha * I).dot(X_train.T).dot(y_train)

            val_predictions = X_val.dot(weights)
            val_score = np.mean((val_predictions - y_val) ** 2)
            mse.append(val_score)

            if val_score > best_score:
                best_score = val_score
                best_weights = weights

        self.weights = best_weights
        self.mse = np.array(mse)
        self.rmse = self.mse**0.5

    def predict(self, X):
        if self.weights is None:
            raise ValueError("Model has not been trained yet. Call the 'fit' method first.")

        X = np.column_stack([np.ones(X.shape[0]), X])
        predictions = X.dot(self.weights)
        return predictions
