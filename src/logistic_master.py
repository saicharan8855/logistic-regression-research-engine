import numpy as np
import pandas as pd

class LogisticRegression():
    
    def __init__(self):
        self.theta = None
        self.loss_history = []
        
    def sigmoid(self,z):
        z = np.asarray(z)
        z = np.clip(z, -500, 500)

        sigmoid_value = 1 / (1 + np.exp(-z))

        return sigmoid_value

    def compute_cost(self,X,y,theta , lambda_reg =  0.0 , penalty = None , l1_ratio = 0.5):
        X = np.asarray(X)
        y = np.asarray(y)
        theta = np.asarray(theta)
        
        m = y.shape[0]
        z = X@theta

        p = self.sigmoid(z)
        epsilon = 1e-15
        p = np.clip(p , epsilon , 1-epsilon)

        cost = -1/m * np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

        if penalty == 'l2':
            cost += (lambda_reg / (2*m) * np.sum[1:]**2)
        elif penalty == 'l1':
            cost += (lambda_reg / m)* np.sum(np.abs(theta[1:]))
        elif penalty == 'elasticnet' :
            l2_term = (lambda_reg / (2 * m)) * np.sum(theta[1:] ** 2)
            l1_term = (lambda_reg / m) * np.sum(np.abs(theta[1:]))
            cost += l1_ratio * l1_term + (1 - l1_ratio) * l2_term
            

        return cost

    def compute_gradient(self,X,y , theta):
        X = np.asarray(X)
        y = np.asarray(y)
        theta = np.asarray(theta)

        m = y.shape[0]
        z = X@theta

        p = self.sigmoid(z)
        gradient = (1/m)*(X.T @ (p-y))

        return gradient

    def compute_hessian(self,X,theta):

        X = np.asarray(X)
        theta = np.asarray(theta)

        m = X.shape[0]

        z = X @ theta

        p = self.sigmoid(z)

        r = (p * (1 - p)).flatten()
        R = np.diag(r)

        H = (1/m) * (X.T @ R @ X)

        return H

    def fit_gd(self , X , y , alpha = 0.01 , epochs = 1000 , lambda_reg = 0.0, penalty = None, l1_ratio = 0.5):
        X = np.asarray(X)
        y = np.asarray(y).reshape(-1, 1)

        m , n = X.shape

        self.theta = np.zeros((n , 1))
        self.loss_history = []

        for _ in range(epochs):
            cost = self.compute_cost(X , y , self.theta)
            self.loss_history.append(cost)

            gradient = self.compute_gradient(X , y , self.theta)

            if penalty == 'l2':
                reg_term = (lambda_reg / m) * self.theta
                reg_term[0] = 0
                gradient += reg_term

            elif penalty == 'l1':
                reg_term = (lambda_reg / m) * np.sign(self.theta)
                reg_term[0] = 0
                gradient += reg_term

            elif penalty == 'elasticnet':
                l2_term = (lambda_reg / m) * self.theta
                l1_term = (lambda_reg / m) * np.sign(self.theta)
                reg_term = l1_ratio * l1_term + (1 - l1_ratio) * l2_term
                reg_term[0] = 0
                gradient += reg_term

            self.theta = self.theta - alpha * gradient
        return self

    def fit_newton(self, X, y, max_iter=100, tol=1e-6 , lambda_reg = 0.0, penalty = None, l1_ratio = 0.5):

        X = np.asarray(X)
        y = np.asarray(y).reshape(-1, 1)

        m, n = X.shape

        self.theta = np.zeros((n, 1))
        self.loss_history = []

        for _ in range(max_iter):

            cost = self.compute_cost(X, y, self.theta)
            self.loss_history.append(cost)

            gradient = self.compute_gradient(X, y, self.theta)
            if penalty == 'l2':
                reg_term = (lambda_reg / m) * self.theta
                reg_term[0] = 0
                gradient += reg_term
                
            elif penalty == 'l1':
                reg_term = (lambda_reg / m) * np.sign(self.theta)
                reg_term[0] = 0
                gradient += reg_term
                
            elif penalty == 'elasticnet':
                l2_term = (lambda_reg / m) * self.theta
                l1_term = (lambda_reg / m) * np.sign(self.theta)
                reg_term = l1_ratio * l1_term + (1 - l1_ratio) * l2_term
                reg_term[0] = 0
                gradient += reg_term


            H = self.compute_hessian(X, self.theta)

        
            self.theta = self.theta - np.linalg.solve(H, gradient)

            if np.linalg.norm(gradient) < tol:
                break

        return self

    def fit_sgd(self, X, y, alpha=0.01, epochs=100):

        X = np.asarray(X)
        y = np.asarray(y).reshape(-1, 1)

        m, n = X.shape

        self.theta = np.zeros((n, 1))
        self.loss_history = []

        for _ in range(epochs):

            cost = self.compute_cost(X, y, self.theta)
            self.loss_history.append(cost)

            for i in range(m):

                X_i = X[i:i+1]      
                y_i = y[i:i+1]

                gradient = self.compute_gradient(X_i, y_i, self.theta)

                self.theta -= alpha * gradient

        return self

    def fit_mini_batch(self, X, y, alpha=0.01, epochs=100, batch_size=32):

        X = np.asarray(X)
        y = np.asarray(y).reshape(-1, 1)

        m, n = X.shape

        self.theta = np.zeros((n, 1))
        self.loss_history = []

        for _ in range(epochs):

            cost = self.compute_cost(X, y, self.theta)
            self.loss_history.append(cost)

            start = 0

            while start < m:

                end = min(start + batch_size, m)

                X_batch = X[start:end]
                y_batch = y[start:end]

                gradient = self.compute_gradient(X_batch, y_batch, self.theta)

                self.theta -= alpha * gradient

                start = end

        return self


    def predict_proba(self, X):

        if self.theta is None:
            raise ValueError("Model not trained yet")

        X = np.asarray(X)

        z = X @ self.theta

        probabilities = self.sigmoid(z)

        return probabilities.flatten()

    def predict(self, X, threshold=0.5):

        probabilities = self.predict_proba(X)

        predictions = (probabilities >= threshold).astype(int)

        return predictions
        







            


        