import numpy as np

class LinearRegression:
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.beta_ = None
        self.Q_ = None
        self.R_ = None

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        if self.fit_intercept:
            ones = np.ones((X.shape[0], 1))
            X = np.hstack((ones, X))

        # QR decomposition: Q = Orthogonal / R = upper triangular
        # Condition number of X^T X is basically worse than QR decompostion
        Q, R = np.linalg.qr(X)
        self.Q_ = Q
        self.R_ = R

        # X^T X beta = X^T y
        # (QR)^T QR beta = (QR)^T y
        # R^T R beta = R^T Q^T y
        self.beta_ = np.linalg.solve(R, Q.T @ y)

        return self

    def predict(self, X):
        X = np.asarray(X)

        if self.fit_intercept:
            ones = np.ones((X.shape[0], 1))
            X = np.hstack((ones, X))

        return X @ self.beta_

    def get_hat_matrix(self):
        # H = X (X^T X)^-1 X^T
        #   = QR ((QR)^T QR)^-1 (QR)^T
        #   = QR (R^T R)^-1 R^T Q^T
        #   = QR R^-1 (R^T)^-1 R^T Q^T
        #   = QQ^T
        return self.Q_ @ self.Q_.T

    def get_summary_stats(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        y_hat = self.predict(X)

        N = X.shape[0]
        K = self.beta_.shape[0] # number of parameters

        TSS = np.sum((y - y.mean()) ** 2)
        RSS = np.sum((y - y_hat) ** 2)
        sigma_squared_hat = RSS / (N - K)
        cov_matrix = sigma_squared_hat * np.linalg.inv(self.R_.T @ self.R_)
        SE = np.sqrt(np.diag(cov_matrix))
        t_values = self.beta_ / SE # H_0_j: beta_j = 0, this indicates data j has little influence

        return t_values, SE

    def get_cooks_distance(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        y_hat = self.predict(X)
        residuals = y - y_hat

        N = X.shape[0]
        K = self.beta_.shape[0]

        RSS = np.sum(residuals ** 2)
        sigma_squared_hat = RSS / (N - K)
        
        H = self.get_hat_matrix()
        Hii = np.diag(H)
        D = residuals ** 2 / (sigma_squared_hat * K) * Hii / (1 - Hii) ** 2
        return D


    def score(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        y_hat = self.predict(X)

        N = X.shape[0]
        K = self.beta_.shape[0]

        TSS = np.sum((y - y.mean()) ** 2)
        RSS = np.sum((y - y_hat) ** 2)

        R2 = 1 - RSS / TSS
        adj_R2 = 1 - (RSS / (N - K)) / (TSS / (N - 1))
        return R2, adj_R2

if __name__ == '__main__':
    model = LinearRegression()

    a = 3
    b = 2
    x = np.linspace(0, 1, 101).reshape(-1, 1)
    eps = np.random.normal(size=x.shape)
    y = a * x + b + eps

    model.fit(x, y)

    print(model.beta_)

    t, s = model.get_summary_stats(x, y)
    print(t)
    print(s)