import numpy as np
from scipy.optimize import fmin_l_bfgs_b

class InformativePriorLogisticRegression:
    """Logistic regression with general spherical Gaussian prior.

    Arguments:
        w0 (ndarray, shape = (n_features,)): coefficient prior
        b0 (float): bias prior
        reg_param (float): regularization parameter $\lambda$ (default: 0)
    """

    def __init__(self, w0=None, b0=0, reg_param=0):
        self.w0 = w0   # prior coefficients
        self.b0 = b0   # prior bias
        self.reg_param = reg_param   # regularization parameter (lambda)
        self.set_params(np.zeros_like(w0), 0)


    def fit(self, X, y):
        """Fit the model according to the given training data.

        Arguments:
            X (ndarray, shape = (n_samples, n_features)):
                Training input matrix where each row is a feature vector.
            y (ndarray, shape = (n_samples,)):
                Training output vector. Each entry is either -1 or 1.
        """        
        # initial values set by set_params when objet was initialised
        _, D = X.shape        
        wb_init = np.zeros(D+1)   # initial guess for weight vector
        w, b = self.get_params()  # set_params inits to zero vector for wb
        wb_init[:-1] = w    # self.w0
        wb_init[-1] = b     # self.b0

        wb_opt, _, _ = fmin_l_bfgs_b(func = self.objective, 
                        x0 = wb_init, 
                        fprime = self.objective_grad, 
                        args = (X,y))
        
        self.set_params(wb_opt[:-1], wb_opt[-1])
        return


    def predict(self, X):
        """Predict class labels for samples in X.

        Arguments:
            X (ndarray, shape = (n_samples, n_features)): test data

        Returns:
            y (ndarray, shape = (n_samples,):
                Predictions with values in {-1, +1}.
        """
        N,_ = X.shape
        y_pred = np.zeros(N)

        for i in range(N):                        
            tmpvar = np.exp(-1 * (np.dot(X[i], self.w) + self.b))
            pr_pos1 = 1.0/(1 + tmpvar)
            pr_neg1 = 1 - pr_pos1
            
            if pr_pos1 >= pr_neg1:
                y_pred[i] = 1
            else:
                y_pred[i] = -1

        return y_pred


    def objective(self, wb, X, y):
        """Compute the objective function

        Arguments:
            wb (ndarray, shape = (n_features + 1,)):
                concatenation of the coefficient and the bias parameters
                wb = [w, b]
            X (ndarray, shape = (n_samples, n_features)):
                Training input matrix where each row is a feature vector.
            y (ndarray, shape = (n_samples,)):
                training label.

        Returns:
            loss (float):
                the objective function evaluated on w.
        """
        N,_ = X.shape
        w = wb[:-1]
        b = wb[-1]
        loss = 0.0  # objective function value
        loss += self.reg_param * np.dot(b-self.b0, b-self.b0)
        loss += self.reg_param * np.dot(w-self.w0, w-self.w0)

        for i in range(N):
            tmpvar = np.exp(-1 * y[i] * (np.dot(w, X[i]) + b))
            loss += np.log(1 + tmpvar)
        
        return loss


    def objective_grad(self, wb, X, y):
        """Compute the derivative of the objective function

        Arguments:
            wb (ndarray, shape = (n_features + 1,)):
                concatenation of the coefficient and the bias parameters
                wb = [w, b]
            X (ndarray, shape = (n_samples, n_features)):
                Training input matrix where each row is a feature vector.
            y (ndarray, shape = (n_samples,)):
                training label.

        Returns:
            loss_grad (ndarray, shape = (n_features + 1,)):
                derivative of the objective function with respect to w.
        """
        N, D = X.shape
        w = wb[:-1]
        b = wb[-1]
        loss_grad = np.zeros(D+1)        
        # grad wrt regularization
        loss_grad[-1] = 2 * self.reg_param * (b - self.b0)  # grad_b
        loss_grad[:-1] = 2 * self.reg_param * (w - self.w0) # grad_w

        for i in range(N):
            tmpvar = np.exp(-1 * y[i] * (np.dot(w, X[i]) + b))              
            loss_grad[-1] += tmpvar/(1 + tmpvar) * -1 * y[i] # grad_b            
            loss_grad[:-1] += tmpvar/(1 + tmpvar) * -1 * y[i] * X[i] # grad_w

        return loss_grad


    def get_params(self):
        """Get parameters for the model.

        Returns:
            A tuple (w,b) where w is the learned coefficients (ndarray)
            and b is the learned bias (float).
        """
        return self.w, self.b


    def set_params(self, w, b):
        """ Set the parameters of the model.

        Arguments:
            w (ndarray, shape = (n_features,)): coefficient prior
            b (float): bias prior
            reg_param (float): regularization parameter $\lambda$ (default: 0)
        """
        self.w = w
        self.b = b
        return 


def main():
    from matplotlib import pyplot as plt
    np.random.seed(0)

    train_X = np.load('../data/q2_train_X.npy')
    train_y = np.load('../data/q2_train_y.npy')
    test_X = np.load('../data/q2_test_X.npy')
    test_y = np.load('../data/q2_test_y.npy')
    w0 = np.load('../data/q2_w_prior.npy').squeeze()
    b0 = np.load('../data/q2_b_prior.npy')

    def test_accuracy(obj, X, y):
        N, _ = X.shape
        y_pred = obj.predict(X)        
        num_correct = sum(y_pred == y)
        accuracy = num_correct/N        
        return accuracy
    
    num_vals = np.arange(10, 410, 10)
    acc_l0 = np.zeros(len(num_vals))
    acc_l10 = np.zeros(len(num_vals))
    
    i = 0   # counter for the training iterations
    for num in num_vals:
        X_tmp_trn = train_X[:num, :]
        y_tmp_trn = train_y[:num]
        
        obj_l0 = InformativePriorLogisticRegression(w0, b0, reg_param=0)
        obj_l0.fit(X_tmp_trn, y_tmp_trn)
 
        obj_l10 = InformativePriorLogisticRegression(w0, b0, reg_param=10)        
        obj_l10.fit(X_tmp_trn, y_tmp_trn)
 
        acc_l0[i] = test_accuracy(obj_l0, test_X, test_y)
        acc_l10[i] = test_accuracy(obj_l10, test_X, test_y)
        i += 1
    

    plt.figure()
    plt.plot(num_vals, acc_l0, 'b-', label='reg=0')
    plt.plot(num_vals, acc_l10, 'r-', label='reg=10')
    plt.legend(loc='upper left')
    plt.title('Accuracy Plot')
    plt.xlabel('Training Examples')
    plt.ylabel('Test Accuracy')
    plt.savefig('../q2.png')

    return


if __name__ == '__main__':
    main()