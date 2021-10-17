import numpy as np
import gzip
import pickle
# from sklearn.linear_model import LogisticRegression



class SVM:
    """SVC with subgradient descent training.

    Arguments:
        C: regularization parameter (default: 1)
        iterations: number of training iterations (default: 500)
    """
    def __init__(self, C=1, iterations=2000):
        self.C = C
        self.iterations = iterations


    def fit(self, X, y):
        """Fit the model using the training data.

        Arguments:
            X (ndarray, shape = (n_samples, n_features)):
                Training input matrix where each row is a feature vector.
            y (ndarray, shape = (n_samples,)):
                Training target. Each entry is either -1 or 1.
        """
        _, D = X.shape        
        # weight init (Or should it be random?)
        w = np.zeros(D)
        b = 0.0
        self.set_model(w, b)

        iters = self.iterations
        lr = 1e-3

        for e in range(1, iters+1):
            # loss = self.objective(X,y)
            w, b = self.get_model()
            grad_w, grad_b = self.subgradient(X,y)           
            w += -1 * lr * grad_w
            b += -1 * lr * grad_b
            self.set_model(w, b)
            # if e % 100 == 0:
            #     print(e, loss)

        # self.set_model(w, b)
        # return w, b
        

    def objective(self, X, y):
        """Compute the objective function for the SVM.

        Arguments:
            X (ndarray, shape = (n_samples, n_features)):
                Training input matrix where each row is a feature vector.
            y (ndarray, shape = (n_samples,)):
                Training target. Each entry is either -1 or 1.

        Returns:
            obj (float): value of the objective function evaluated on X and y.
        """
        w, b = self.get_model()
        N,_ = X.shape
        loss = 0.0
        loss += np.dot(w,w)        
        for i in range(N):
            diff = 1 - y[i] * (b + np.dot(w, X[i]))
            loss += self.C * max(0, diff)
        
        return loss


    def subgradient(self, X, y):
        """Compute the subgradient of the objective function.

        Arguments:
            X (ndarray, shape = (n_samples, n_features)):
                Training input matrix where each row is a feature vector.
            y (ndarray, shape = (n_samples,)):
                Training target. Each entry is either -1 or 1.

        Returns:
            subgrad_w (ndarray, shape = (n_features,)):
                subgradient of the objective function with respect to
                the coefficients of the linear model.
            subgrad_b (float):
                subgradient of the objective function with respect to
                the bias term.
        """
        N, D = X.shape
        # Non-data dependent terms of the gradient
        w, b = self.get_model()
        subgrad_w = 2 * w
        subgrad_b = 0.0
        for i in range(N):
            diff = 1 - y[i] * (b + np.dot(w, X[i]))
            if diff > 0:
                subgrad_w += self.C * -1 * y[i] * X[i]
                subgrad_b += self.C * -1 * y[i]
            # elif np.abs(diff) < 1e-6:   # i.e diff == 0.0
            #     subgrad_w += self.C * ???
            #     subgrad_b += self.C * ???
        
        return subgrad_w, subgrad_b


    def predict(self, X):
        """Predict class labels for samples in X.

        Arguments:
            X (ndarray, shape = (n_samples, n_features)): test data

        Returns:
            y (ndarray, shape = (n_samples,):
                Predictions with values of -1 or 1.
        """
        N, D = X.shape
        y = np.zeros(N)
        w, b = self.get_model()
        for i in range(N):
            tmp = b + np.dot(w, X[i])
            if tmp >= 0:    # WHAT TO DO AT TMP==0 ???
                y[i] = 1
            else:
                y[i] = -1
        
        return y


    def get_model(self):
        """Get the model parameters.

        Returns:
            w (ndarray, shape = (n_features,)):
                coefficient of the linear model.
            b (float): bias term.
        """
        return self.w, self.b


    def set_model(self, w, b):
        """Set the model parameters.

        Arguments:
            w (ndarray, shape = (n_features,)):
                coefficient of the linear model.
            b (float): bias term.
        """
        self.w, self.b = w, b


def main():
    np.random.seed(0)

    with gzip.open('../data/svm_data.pkl.gz', 'rb') as f:
        train_X, train_y, test_X, test_y = pickle.load(f)

    def classification_acc(model, test_X, test_y):
        N,_ = test_X.shape
        y_pred = model.predict(test_X)
        count = sum(y_pred == test_y)
        accuracy = count/N
        return accuracy

    def compute_logis_loss(X, y, w, b):
        N,_ = X.shape
        loss = 0.0  # objective function value
        # loss += self.reg_param * np.dot(b-self.b0, b-self.b0)
        # loss += self.reg_param * np.dot(w-self.w0, w-self.w0)
        for i in range(N):
            tmpvar = np.exp(-1 * y[i] * (np.dot(w, X[i]) + b))
            loss += np.log(1 + tmpvar)            
        return loss

    cls = SVM()
    cls.fit(train_X, train_y)
    # w,b = cls.get_model()
    svm_loss = cls.objective(train_X, train_y)
    svm_acc = classification_acc(cls, train_X, train_y)
    print("SVM Loss: ", svm_loss)
    print("SVM accuracy: ", svm_acc)

    cl_logis = LogisticRegression(solver='lbfgs')
    cl_logis.fit(train_X, train_y)

    w_l = cl_logis.coef_.flatten()
    b_l = cl_logis.intercept_.flatten()

    logis_acc = cl_logis.score(train_X, train_y)
    logis_loss = compute_logis_loss(train_X, train_y, w_l, b_l)
    print("Scikit LogisticRegression loss: ", logis_loss)
    print("Scikit LogisticRegression accuracy: ", logis_acc)
    
    svm_tmp = SVM()
    svm_tmp.set_model(w_l, b_l)
    svm2_loss = svm_tmp.objective(train_X, train_y)
    svm2_acc = classification_acc(svm_tmp, train_X, train_y)
    print("Svm loss with params from LogisReg model: ", svm2_loss)
    print("Svm accuracy with params from LogisReg model: ", svm2_acc)

    svm_loss_test = cls.objective(test_X, test_y)
    svm_acc_test = classification_acc(cls, test_X, test_y)
    print("SVM Loss: ", svm_loss_test)
    print("SVM accuracy on test data: ", svm_acc_test)

    # cl_logis = LogisticRegression(solver='lbfgs')
    # cl_logis.fit(train_X, train_y)

    # w_l = cl_logis.coef_.flatten()
    # b_l = cl_logis.intercept_.flatten()

    logis_acc_test = cl_logis.score(test_X, test_y)
    logis_loss_test = compute_logis_loss(test_X, test_y, w_l, b_l)
    print("Scikit LogisticRegression loss: ", logis_loss_test)
    print("Scikit LogisticRegression accuracy on test data: ", logis_acc_test) 




    
if __name__ == '__main__':
    main()
