import pickle
from blr import *


def util_plot_2a(trn_X, trn_Y, m_vals, dict_samples):

    for i, m in enumerate(m_vals):
        print('Begin for mval: ', m)
        m_samples = dict_samples[m]
        d_X = trn_X[:m]
        d_Y = trn_Y[:m]
        dX1 = d_X[d_Y == 1]
        dX0 = d_X[d_Y == 0]


        plt.figure()
        for theta in m_samples:
            plot_boundary(theta)
        
        plt.plot(dX0[:,0], dX0[:,1], 'ro', label='class-0')
        plt.plot(dX1[:,0], dX1[:,1], 'go', label='class-1')
        
        plt.xlim((-1,1))
        plt.ylim((-1,1))
        
        plt.xlabel('x1')
        plt.ylabel('x2')
        
        plt.title('Q2a ALL: M = ' + str(m))
        plt.legend(loc='upper left')
        # plt.show()
        fname = '../out/plot_2a_m-' + str(m) + '.png'
        plt.savefig(fname)
    
    return


def util_plot_2b(trn_X, trn_Y, m_vals, dict_samples):

    for i, m in enumerate(m_vals):
        print('Begin for mval: ', m)
        m_samples = dict_samples[m]
        d_X = trn_X[:m]
        d_Y = trn_Y[:m]
        dX1 = d_X[d_Y == 1]
        dX0 = d_X[d_Y == 0]
        theta_mean = np.mean(m_samples, axis=0)
        
        plt.figure()
        plot_boundary(theta_mean)        
        plt.plot(dX0[:,0], dX0[:,1], 'ro', label='class-0')
        plt.plot(dX1[:,0], dX1[:,1], 'go', label='class-1')
        
        plt.xlim((-1,1))
        plt.ylim((-1,1))
        
        plt.xlabel('x1')
        plt.ylabel('x2')
        
        plt.title('Q2b AVG: M = ' + str(m))
        plt.legend(loc='upper left')
        # plt.show()
        fname = '../out/plot_2b_m-' + str(m) + '.png'
        plt.savefig(fname)
    
    return


def util_plot_2c(trn_X, trn_Y, m_vals):

    for i, m in enumerate(m_vals):
        print('Begin for mval: ', m)
        # m_samples = dict_samples[m]
        d_X = trn_X[:m]
        d_Y = trn_Y[:m]
        dX1 = d_X[d_Y == 1]
        dX0 = d_X[d_Y == 0]
        
        theta_map = util_estimate_scikit(d_X, d_Y, 200) # C = 200 => MAP
        
        plt.figure()
        plot_boundary(theta_map)        
        plt.plot(dX0[:,0], dX0[:,1], 'ro', label='class-0')
        plt.plot(dX1[:,0], dX1[:,1], 'go', label='class-1')
        
        plt.xlim((-1,1))
        plt.ylim((-1,1))
        
        plt.xlabel('x1')
        plt.ylabel('x2')
        
        plt.title('Q2c MAP: M = ' + str(m))
        plt.legend(loc='upper left')
        # plt.show()
        fname = '../out/plot_2c_m-' + str(m) + '.png'
        plt.savefig(fname)
    
    return


def util_plot_2d(trn_X, trn_Y, m_vals):
    for i, m in enumerate(m_vals):
        print('Begin for mval: ', m)
        # m_samples = dict_samples[m]
        d_X = trn_X[:m]
        d_Y = trn_Y[:m]
        dX1 = d_X[d_Y == 1]
        dX0 = d_X[d_Y == 0]        
        theta_mle = util_estimate_scikit(d_X, d_Y, 1e11)    # C = 1e11 => MLE
        
        plt.figure()
        plot_boundary(theta_mle)        
        plt.plot(dX0[:,0], dX0[:,1], 'ro', label='class-0')
        plt.plot(dX1[:,0], dX1[:,1], 'go', label='class-1')
        
        plt.xlim((-1,1))
        plt.ylim((-1,1))
        
        plt.xlabel('x1')
        plt.ylabel('x2')
        
        plt.title('Q2d MLE: M = ' + str(m))
        plt.legend(loc='upper left')
        # plt.show()
        fname = '../out/plot_2d_m-' + str(m) + '.png'
        plt.savefig(fname)
    
    return


def util_plot_2e(trn_X, trn_Y, m_vals, dict_samples):
    
    for i, m in enumerate(m_vals):
        print('Begin for mval: ', m)
        m_samples = dict_samples[m]
        d_X = trn_X[:m]
        d_Y = trn_Y[:m]
        dX1 = d_X[d_Y == 1]
        dX0 = d_X[d_Y == 0]
        
        theta_avg = np.mean(m_samples, axis=0)
        theta_map = util_estimate_scikit(d_X, d_Y, 200) # C = 200 => MAP
        theta_mle = util_estimate_scikit(d_X, d_Y, 1e11)    # C = 1e11 => MLE

        plt.figure()
        # ~ for theta in m_samples:
            # ~ plt.scatter(theta[0], theta[1], c='lightcyan')
        
        plt.scatter(m_samples[:,0], m_samples[:,1], c='lightcyan', label='samples')
        plt.scatter(theta_avg[0], theta_avg[1], c='blue', label='avg')
        plt.scatter(theta_map[0], theta_map[1], c='red', label='map')
        plt.scatter(theta_mle[0], theta_mle[1], c='green', label='mle')
        
        plt.xlim((0,60))
        plt.ylim((-15,25))
        
        plt.xlabel('w1')
        plt.ylabel('w2')
        
        plt.title('Q2e Weight space plot: M = ' + str(m))
        plt.legend(loc='lower right')
        # plt.show()
        fname = '../out/plot_2e_m-' + str(m) + '.png'
        plt.savefig(fname)
    
    return


def util_comparsion(tst_X, tst_Y, data_X, data_Y, samples_theta):
  '''
  Returns the four estimates required L_(bay, mean, map, mle) for the test data
  i.e the log-likelihood of the test data
  '''
  results = np.zeros(4) # 4 values to compute: Bay, Mean, MLE, MAP
  N, _ = tst_X.shape

  theta_avg = np.mean(samples_theta, axis=0)
  theta_map = util_estimate_scikit(data_X, data_Y, 200) # C = 200 => MAP
  theta_mle = util_estimate_scikit(data_X, data_Y, 1e11)  # C = 1e11 => MLE  

  L_bay = 0.0
  L_mean = 0.0
  L_map = 0.0
  L_mle = 0.0

  for i in range(N):
    print('Current test example number: ', i+1)
    pred_p = predictive_distribution(tst_X[i], tst_Y[i], samples_theta)
    L_bay += np.log(pred_p) 
    L_mean += util_log_prob(tst_X[i], tst_Y[i], theta_avg)
    L_map += util_log_prob(tst_X[i], tst_Y[i], theta_map)
    L_mle += util_log_prob(tst_X[i], tst_Y[i], theta_mle)        
    # results += np.array([L_bay, L_mean, L_map, L_mle])
  results =  np.array([L_bay, L_mean, L_map, L_mle])
  return results


def util_plot_trend_lines(tst_X, tst_Y, trn_X, trn_Y, m_vals, dict_samples):  
  result_mat = np.zeros((len(m_vals), 4)) # total 4 values to store  

  for i, m in enumerate(m_vals):
    print('Begin for mval: ', m)
    d_X = trn_X[:m]
    d_Y = trn_Y[:m]
    m_samples = dict_samples[m]
    tmp = util_comparsion(tst_X, tst_Y, d_X, d_Y, m_samples)
    # tmp gives L_bay, mean, map, mle
    result_mat[i] = tmp
    print('\n')

  # Create the plot
  plt.figure()
  plt.plot(m_vals, result_mat[:, 0], 'bo-', label='Bay-inf')
  plt.plot(m_vals, result_mat[:, 1], 'go-', label='Mean')
  plt.plot(m_vals, result_mat[:, 2], 'ro-', label='MAP')
  plt.plot(m_vals, result_mat[:, 3], 'yo-', label='MLE')
  plt.title('Trend line plot for test set log-likelihood')
  plt.xlabel('number of trainig cases')
  plt.ylabel('log likelihood')
  plt.legend(loc='lower right')
  # plt.show()
  plt.savefig('../out/plot_3b.png')  

  return result_mat


def util_store_samples(S, trn_X, trn_Y, mu_0, cov_0, m_vals):
  # S = 100
  outfname = '../out/samples_s' + str(S) + '.pk'
  outdict = {}

  for m in m_vals:
    d_X = trn_X[:m]
    d_Y = trn_Y[:m]
    outdict[m] = rejection_sampler(S, d_X, d_Y, mu_0, cov_0)
    print('stored samples for m = ', m)
  
  with open(outfname, 'wb') as ofile:
    pickle.dump(outdict, ofile)

  return outdict




def main():
  data = np.load("../data/data.npz")
  Xte = data["Xte"] #Test feature vecotrs
  Yte = data["Yte"] #Test labels
  Xtr = data["Xtr"] #Train feature vectors
  Ytr = data["Ytr"] #Train labels

  mu = np.zeros(3)
  cov = 100 * np.identity(3)
  mvals = [10, 30, 50]
  samp_dict = util_store_samples(100, Xtr, Ytr, mu, cov, mvals)
  # with open('../out/samples_s100.pk', 'rb') as rfile:
    # samp_dict = pickle.load(rfile)  

  util_plot_2a(Xtr, Ytr, mvals, samp_dict)
  util_plot_2b(Xtr, Ytr, mvals, samp_dict)
  util_plot_2c(Xtr, Ytr, mvals)
  util_plot_2d(Xtr, Ytr, mvals)
  util_plot_2e(Xtr, Ytr, mvals, samp_dict)
  util_plot_trend_lines(Xte, Yte, Xtr, Ytr, mvals, samp_dict)


if __name__ == "__main__":
  main()
