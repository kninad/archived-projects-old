import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


def util_log_prob(x, y, theta):
  ''' Utility to calculate the log probability for logistic regression
  Args:
    x (numpy array): the feature vector
    y (int): binary (0 or 1) indicating the class
    theta (numpy array): model parameters, shape=(3,)

  Returns:
    log_prob (float): log of the probability P(y | x, theta)
  '''
  w = theta[:-1]
  b = theta[-1] 
  com_term = np.exp(-1 * (np.dot(w, x) + b))  
  if y == 1:
    log_prob = -1 * np.log(1 + com_term)
    return log_prob
  elif y == 0:
    log_prob = np.log(com_term) - np.log(1 + com_term)
    return log_prob


def util_log_likeihood(data_X, data_Y, theta):
  ''' Utility to calculate the log likelihood of data given the params theta.
  Args:
    data_X (ndarray): Feature vectors from the training data
    data_Y (int array): Binary labeles
    theta (np array): model params

  Returns:
    log_sum (float): log likelihood of the data
  '''  
  N,_ = data_X.shape
  log_sum = 0.0
  for i in range(N):
    log_sum += util_log_prob(data_X[i], data_Y[i], theta)
  return log_sum



def util_estimate_scikit(data_X, data_Y, cval):
  ''' Utility to calculate to fit a model using scikit-learn LogisticRegression
  and get its model params.
  Args:
    data_X (ndarray): Feature vectors from the training data
    data_Y (int array): Binary labeles
    cval (float): inverse of the regularization strength for the model

  Returns:
    theta_estimate (np array): MLE or MAP estimate from the model
  '''  
  # clf = LogisticRegression(tol=1e-2, C=cval, solver='lbfgs')
  clf = LogisticRegression(tol=1e-2, C=cval)
  clf.fit(data_X, data_Y)  
  theta_mle = np.zeros(3)
  theta_mle[:-1] = clf.coef_
  theta_mle[-1] = clf.intercept_
  # print("Estimate and C: ", theta_mle, cval)
  return theta_mle





def rejection_sampler(num_samples, data_X, data_Y, mu_0, cov_0):
  ''' Rejection sampling using Smith-Gelfand algorithm
  Args:
    num_samples (int): number of samples required
    data_X (ndarray): Feature vectors from the training data
    data_Y (int array): Binary labeles
    mu_0 (ndarray): mean vector for prior
    cov_0 (ndarray): covariance matrix for the prior

  Returns:
    samples (ndarray): required samples for theta, shape=(num_samples, 3)
  '''
  samples = np.zeros((num_samples, 3))  # since each theta is (3,)
  theta_mle = util_estimate_scikit(data_X, data_Y, cval=1e11) # C=1e11 => MLE
  log_Pmle = util_log_likeihood(data_X, data_Y, theta=theta_mle)

  for i in range(num_samples):    
    flag_accept = False
    ret_theta = None
    while not flag_accept:
      theta_s = np.random.multivariate_normal(mu_0, cov_0)
      log_Ps = util_log_likeihood(data_X, data_Y, theta=theta_s)
      log_ratio = log_Ps - log_Pmle   # ratio = P_s/P_mle
      u = np.random.rand()        
      # since log is a strictly increasing function:
      if log_ratio >= np.log(u):
        ret_theta = theta_s
        flag_accept = True
        break
        
    samples[i] = ret_theta    
    print('Sample number: ', i+1)     
  return samples


def plot_boundary(theta):
  ''' Function to plot out the decision boundary for theta. 
  Args:
    theta (ndarray): Model paramters (weight vector and bias)

  Returns:
    None
  '''  
  w = theta[:-1]
  b = theta[-1]
  assert np.abs(w[1] - 0) >= 1e-3
  a = -1 * w[0]/w[1]
  c = -1 * b/w[1]
  x1_vec = np.linspace(-1,1)
  x2_vec = a * x1_vec + c
  plt.plot(x1_vec, x2_vec, ls='-', color='lightcyan')  
  return


def predictive_distribution(x, y, samples_theta):   
  ''' Function to calculate the predictive distribution, using a samples of
  theta from the posterior.
  Args:
    x (ndarray): Feature vector
    y (int): Binary label
    samples_theta (ndarray): samples of theta(model params) from the posterior

  Returns:
    mc_sum (float): monte carlo estimate of the predictive probability.
  '''
  S,_ = samples_theta.shape # number of samples considered
  mc_sum = 0.0 
  for i in range(S):    
    logprob = util_log_prob(x, y, samples_theta[i])
    mc_sum += np.exp(logprob)
  mc_sum /= S   # get the average value
  return mc_sum


# def main():
#   data = np.load("../data/data.npz")
#   Xte = data["Xte"] #Test feature vecotrs
#   Yte = data["Yte"] #Test labels
#   Xtr = data["Xtr"] #Train feature vectors
#   Ytr = data["Ytr"] #Train labels

# if __name__ == "__main__":
#   main()
