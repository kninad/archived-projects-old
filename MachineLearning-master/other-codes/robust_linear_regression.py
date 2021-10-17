import numpy as np
from scipy.optimize import fmin_l_bfgs_b

class RobustLinearRegression:
    """Generalized robust linear regression.

    Arguments:
        delta (float): the cut-off point for switching to linear loss
        k (float): parameter controlling the order of the polynomial part of
            the loss
    """

    def __init__(self, delta, k):
        self.delta = delta   # cut-off point
        self.k = k    # polynomial order parameter

    def fit(self, X, y):
        """Fit the model according to the given training data.

        Arguments:
            X (ndarray, shape = (n_samples, n_features)):
                Training input matrix where each row is a feature vector.
            y (ndarray, shape = (n_samples,)):
                Training output vector. Each entry is either -1 or 1.
        """
        D = X.shape[-1]
        wb_init = np.zeros(D+1)
        wb_opt, _, _ = fmin_l_bfgs_b(func = self.objective, 
                        x0 = wb_init, 
                        fprime = self.objective_grad, 
                        args = (X,y))
        
        self.set_params(wb_opt[:-1], wb_opt[-1])
        return

    def predict(self, X):
        """Predict using the linear model.

        Arguments:
            X (ndarray, shape = (n_samples, n_features)): test data

        Returns:
            y (ndarray, shape = (n_samples,): predicted values
        """
        return X.dot(self.w) + self.b

    def objective(self, wb, X, y):
        """Compute the loss function.

        Arguments:
            wb (ndarray, shape = (n_features + 1,)):
                concatenation of the coefficient and the bias parameters
                wb = [w, b]
            X (ndarray, shape = (n_samples, n_features)):
                Training input matrix where each row is a feature vector.
            y (ndarray, shape = (n_samples,)):
                target values.

        Returns:
            loss (float):
                the objective function evaluated on w.
        """
        loss = 0.0
        N, _ = X.shape
        w = wb[:-1]
        b = wb[-1]

        for i in range(N):
            y_pred = np.dot(w, X[i]) + b
            diff = y[i] - y_pred

            if np.abs(diff) <= self.delta:
                loss += 1.0/(2*self.k) * diff**(2*self.k)
            else:
                tmp = np.abs(diff) - (2*self.k - 1)/(2*self.k) * self.delta                
                loss += np.power(self.delta, 2*self.k - 1) *  tmp
        
        return loss

    def objective_grad(self, wb, X, y):
        """Compute the derivative of the loss function.

        Arguments:
            wb (ndarray, shape = (n_features + 1,)):
                concatenation of the coefficient and the bias parameters
                wb = [w, b]
            X (ndarray, shape = (n_samples, n_features)):
                Training input matrix where each row is a feature vector.
            y (ndarray, shape = (n_samples,)):
                target values.

        Returns:
            loss_grad (ndarray, shape = (n_features + 1,)):
                derivative of the objective function with respect to w.
        """
        N, _ = X.shape
        dim = wb.shape
        loss_grad = np.zeros(dim)
        w = wb[:-1]
        b = wb[-1]

        for i in range(N):
            y_pred = np.dot(w, X[i]) + b
            diff = y[i] - y_pred

            if np.abs(diff) <= self.delta:
                loss_grad[-1] += diff**(2*self.k - 1) * -1  # grad_b
                loss_grad[:-1] += diff**(2*self.k - 1) * -1 * X[i]  # grad_w
            else:
                loss_grad[-1] += self.delta**(2*self.k - 1) * np.sign(diff) * -1
                loss_grad[:-1] += self.delta**(2*self.k - 1) * np.sign(diff) * -1 * X[i]

        return loss_grad

    def get_params(self):
        """Get learned parameters for the model. Assumed to be stored in
           self.w, self.b.

        Returns:
            A tuple (w,b) where w is the learned coefficients (ndarray)
            and b is the learned bias (float).
        """
        return self.w, self.b

    def set_params(self, w, b):
        """Set the parameters of the model. When called, this
           function sets the model parameters tha are used
           to make predictions. Assumes parameters are stored in
           self.w, self.b.

        Arguments:
            w (ndarray, shape = (n_features,)): coefficient prior
            b (float): bias prior
        """
        self.w = w
        self.b = b


def main():
    from sklearn.metrics import mean_squared_error
    from sklearn import linear_model
    from matplotlib import pyplot as plt

    np.random.seed(0)

    train_X = np.load('../data/q3_train_X.npy')
    train_y = np.load('../data/q3_train_y.npy')

    obj = RobustLinearRegression(delta=1, k=1)
    obj.fit(train_X, train_y)
    y_pred = obj.predict(train_X)
    mse_robust = mean_squared_error(y_pred, train_y)

    scikit_linmodel = linear_model.LinearRegression()
    scikit_linmodel.fit(train_X, train_y)
    y_pred_scikit = scikit_linmodel.predict(train_X)
    mse_scikit = mean_squared_error(y_pred_scikit, train_y)

    print('--------Reporting MSE scores--------')
    print("Robust Linear Regression:", mse_robust)    
    print("Scikit standard linear model:", mse_scikit)

    w_mod, b_mod = obj.get_params()
    w_sci = scikit_linmodel.coef_
    b_sci = scikit_linmodel.intercept_

    plt.figure()
    plt.plot(train_X, train_y, 'bo', label='Scatter plot')
    axes = plt.gca()

    x_vals_1 = np.array(axes.get_xlim())
    y_vals_1 = b_mod + w_mod * x_vals_1
    plt.plot(x_vals_1, y_vals_1, 'r-', label='Robust Model')

    x_vals_2 = np.array(axes.get_xlim())
    y_vals_2 = b_sci + w_sci * x_vals_2
    plt.plot(x_vals_2, y_vals_2, 'g-', label='Standard Model')

    plt.title('Scatter Plot with Regression lines')
    plt.xlabel('x values')
    plt.ylabel('y values')
    plt.legend(loc='upper left')
    plt.savefig('../q3.png')

if __name__ == '__main__':
    main()
