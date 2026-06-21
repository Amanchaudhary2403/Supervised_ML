import numpy as np

class Linear_Regression:
    def __init__(self, regularisation_l2=False, alpha=1.0):
        self.regularisation = regularisation_l2
        self.alpha = alpha

        self.coeff = None
        self.intercept = 0.0

    def fit(self, X_, y_):

        X_ = np.asarray(X_, dtype=np.float64)
        y_ = np.asarray(y_, dtype=np.float64)

        # Make them Zero-centered to calculate intercept
        X_mean = np.mean(X_, axis = 0)
        y_mean = np.mean(y_, axis = 0)
        
        X = X_ - X_mean
        y = y_ - y_mean

        if y.ndim == 1:
            y = y.reshape(-1, 1)

        # theta = (X.T @ X)-1 @ X.T @ y
        # With l2: theta = (X.T @ X + I * alpha)-1 @ X.T @ y

        XtX = X.T @ X

        if self.regularisation:
            n_fet = XtX.shape[0]
            I = np.eye(n_fet)

            XtX = XtX + self.alpha * I

        inv_mat = np.linalg.pinv(XtX)

        theta = inv_mat @ X.T @ y

        self.coeff = theta.flatten()

        # b = y_mean - X_mean @ theta
        self.intercept = (y_mean - X_mean @ self.coeff)

        return self

    def predict(self ,X):
        X = np.asarray(X, dtype=np.float64)
        return X @ self.coeff + self.intercept
      