import numpy as np
from scipy.sparse import load_npz
import pickle

"""
Ninad Khargonkar (nkhargonkar@umass.edu)

TODO
- Re-check the MAE computation code!!!!
- Run the K-fold CV on aws server. Automate the process for K.
  - Split it across two files to run in parallel?
- Check cval calculation in predict-utility
- Vectorize the code where possible to avoid TLE: predict, logpost, fit
"""


class CategorialMixture:
    """Mixture of categorical variables

    Arguments:
        n_dims (int):
            number of categorical variables.
        n_categories (int):
            number of categories in each categorical variable.
        n_components (int):
            number of mixture components.
        n_iter (int):
            number of iterations for the EM algorithm.
    """
    def __init__(self, n_dims, n_categories, n_components, n_iter=20):
        self.dims = n_dims        
        self.catg = n_categories
        self.comp = n_components
        self.iterations = n_iter

        self.theta = None
        self.beta = None
        theta = np.random.dirichlet(np.random.randint(1,10, size=self.comp))
        beta = np.random.dirichlet(np.random.randint(1,10, size=self.catg), self.dims*self.comp)
        beta = beta.reshape(self.dims,self.comp,self.catg)
        beta = np.moveaxis(beta, -1, 0)        
        self.set_model(theta, beta)

    
    # Finds out the log marginal-prob for a single data vector
    # Uses the log-sum-exp trick for numerical stability
    # P(X_o = x_o | theta, beta)
    def util_logmarginal_ind(self, xvec, theta, beta):
        # theta, beta = self.get_model()
        ans = np.zeros(self.comp)
        for d in range(len(xvec)):             
            if xvec[d] != 0:
                c = xvec[d] - 1
                assert c >= 0
                ans += np.log(beta[c, d])
        out = ans + np.log(theta)
        max_out = max(out)
        logprob = max_out + np.log(np.sum(np.exp(out-max_out)))
        return logprob


    # Finds out the log posterior-prob for all zval given a data vector
    # Returns a vector of size (K,) with post-prob for each z
    # P(Z = z | X_o = x_o, theta, beta )
    def util_logpost(self, xvec, theta, beta):
        # theta, beta = self.get_model()
        obs_dims = np.where(xvec>0)[0]
        com_log_denom = self.util_logmarginal_ind(xvec, theta, beta)        

        log_prob_vec = np.log(self.theta)
        for s in obs_dims:
            c = xvec[s] - 1
            assert c >= 0 
            log_prob_vec += np.log(beta[c, s, :])
        
        log_prob_vec -= com_log_denom
        return log_prob_vec


    def fit(self, X):
        """Fit the mixture model using EM algorithm.

        NOTE: In the tests, set_model will be called prior to fit
        to initialize the parameters. The EM algorithm should start with
        those parameters without a separate initialization.

        Arguments:
            X (integer numpy array, shape = (n_samples, n_dims)):
                Data matrix where each row is a feature vector
                with values in {0, 1, 2, 3, 4, 5} where 0 indicates
                a missing entry.
        """
        num_iters = self.iterations
        # theta, beta = self.get_model()
        N,_ = X.shape

        # Create an intermediate array to store the phi_zn values
        # Size = (K, N)
        phiarr = np.zeros((self.comp, N))
        for t in range(num_iters):      

            # First retrieve the previous thetas and betas
            theta, beta = self.get_model()

            # Update the values for all phi_zn 
            for n in range(N):                
                tmp_logprobvec = self.util_logpost(X[n], theta, beta)
                phiarr[:, n] = np.exp(tmp_logprobvec)         
            
            # Write out the EM update for theta
            t_theta = np.sum(phiarr, axis=1)                                    
            t_theta += (1.0/self.comp)            
            t_theta /= (N+1)   
           
           #TODO: think about further optimizing this code! added vectorization
            # Write out the EM updates for beta 3d tensor
            t_beta = np.zeros_like(self.beta)
            for z in range(self.comp):
                phi_z = phiarr[z,:]
                for d in range(self.dims):
                    for c in range(self.catg):                    
                        indic = (X[:,d] == c+1)
                        numr = np.dot(phi_z, indic)
                        numr += (1.0/self.catg)
                        t_beta[c,d,z] = numr
                    lamb = np.sum(t_beta[:,d,z])
                    t_beta[:,d,z] /= lamb
            
            assert t_theta.shape == self.theta.shape
            assert t_beta.shape == self.beta.shape            
            # Set the model params to the current value of theta and beta
            self.set_model(t_theta, t_beta)            
            
            logML = self.log_marginal_likelihood(X)
            print(t+1, logML)

        return
        # raise NotImplementedError


    def log_marginal_likelihood(self, X):
        """Compute the log marginal likelihood of the data.

        Arguments:
            X (integer numpy array, shape = (n_samples, n_dims)):
                Data matrix where each row is a feature vector
                with values in {0, 1, 2, 3, 4, 5} where 0 indicates
                a missing entry.

        Returns:
            log marginal likelihood. (float)
        """
        theta, beta = self.get_model()
        N,_ = X.shape                
        logmarg_lk = 0.0
        for i in range(N):
            logmarg_lk += self.util_logmarginal_ind(X[i], theta, beta)
        return logmarg_lk
        # raise NotImplementedError


    def log_posterior(self, X, z):
        """Compute the log posterior probability of the data given the
        corresponding mixture components.

        Arguments:
            X (integer numpy array, shape = (n_samples, n_dims)):
                Data matrix where each row is a feature vector
                with values in {0, 1, 2, 3, 4, 5} where 0 indicates
                a missing entry.

            z (integer numpy array, shape = (n_samples,)):
                Mixture component for each sample.
                z takes on values in {0, 1, ..., n_components-1}.

        Returns:
            log posterior probability (numpy array, shape = (n_samples,))
        """
        theta, beta = self.get_model()
        N, _ = X.shape
        log_post_array = np.zeros(N)
        assert log_post_array.shape == z.shape

        for i in range(N):
            tmpvec = self.util_logpost(X[i], theta, beta)
            # only need the value for zval=z[i]
            log_post_array[i] = tmpvec[z[i]]

        return log_post_array
        # raise NotImplementedError


    def predict(self, X):
        """Predict missing entries in X.

        Predict and fill in the missing entries.

        Arguments:
            X (integer numpy array, shape = (n_samples, n_dims)):
                Data matrix where each row is a feature vector
                with values in {0, 1, 2, 3, 4, 5} where 0 indicates
                a missing entry.

        Returns:
            X_predict (*float* numpy array, shape = (n_samples, n_dims)):
                X with missing data filled by the predictions.
                The originally observed entries should be kept intact.
        """
        theta, beta = self.get_model()
        N, _ = X.shape
        X_predict = np.copy(X).astype(float)

        for i in range(N):
            xvec = X[i]
            pvec = X_predict[i]
            hid_dims = np.where(xvec == 0)[0]
            # postvec is common to all hidden-dims 
            postvec = np.exp(self.util_logpost(xvec, theta, beta))  # (K,) vector                    
            
            for h in hid_dims:
                cvalvec = np.array([c+1 for c in range(self.catg)])            
                betamat = beta[:, h, :]  # (C,K) matrix
                predvec = np.dot(betamat, postvec)  # (C,) vector
                # predvec is a vector for P(X_h=c | X_o = x_o, params) across all c

                pred_h = np.dot(cvalvec, predvec)  # final expectation (scalar)
                pvec[h] = pred_h
        
        return X_predict
        # raise NotImplementedError


    def get_model(self):
        """Get the model parameters.

        Returns:
            theta (numpy array, shape = (n_components,)):
                Probability vector of the mixture.

            beta (numpy array, shape = (n_categories, n_dims, n_components)):
                Probability vectors of the categorical variables.
        """
        return self.theta, self.beta


    def set_model(self, theta, beta):
        """Set the model parameters.

        Arguments:
            theta (numpy array, shape = (n_components,)):
                Probability vector of the mixture.

            beta (numpy array, shape = (n_categories, n_dims, n_components)):
                Probability vectors of the categorical variables.
        """
        self.theta = theta
        self.beta = beta


####### MAIN LEVEL CODE ##########

def mae_calc(train_X, pred_trn_X, test_X):

    def util_compute_mae(x, y, dims):
    # dims is the hidden dims in x
    # we should go over a dim if y[dim] != 0
        mae = 0
        count = 0
        # dims1 = set(dims)
        # dims2 = set(np.where(y != 0)[0])
        # com_dims = dims1.intersection(dims2)
        # print("Len common dim:", len(com_dims))    
        dims2 = np.where(y != 0)[0]        
        for i in dims:
            if y[i] != 0:
                mae += np.abs(x[i] - y[i])    

        count = len(dims2)
        return mae, count

    N, _ = train_X.shape
    total_mae = 0.0
    total_count = 0

    print("Predict over, Begin computing MAE")
    for i in range(N):
        hid_dims = np.where(train_X[i] == 0)[0]  
        # print(len(hid_dims))
        ind_mae, ind_count = util_compute_mae(pred_trn_X[i], test_X[i], hid_dims)    
        total_mae += ind_mae
        total_count += ind_count
        # print(ind_mae, total_mae, ind_count, total_count, '\n')

    total_mae /= total_count
    print("Total mae", total_mae)

    return total_mae



def main():
    np.random.seed(0)

    train_X = load_npz('../data/train.npz').toarray()
    test_X = load_npz('../data/test.npz').toarray()

    n_categories = len(np.unique(train_X)) - 1   # zero: missing entry
    n_dims = train_X.shape[1]
    mm = CategorialMixture(n_dims, n_categories, n_components=3, n_iter=20)
    mm.fit(train_X)

   
if __name__ == '__main__':
    main()
