import numpy as np 
from matplotlib import pyplot as plt
import os
import sys

def plot_vec(yvec, fname, xlab, ylab, title):
    n = len(yvec)
    xvec = np.arange(1,n+1)
    plt.figure()
    plt.plot(xvec, yvec)    
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title)
    plt.savefig(fname)
    plt.close()

def plot_all_in_logfolder(experiment_folder):
    root = experiment_folder

    evalfile = root + 'evals.txt'
    lossfile = root + 'losses.txt'
    
    # loss_tuple = (loss_value, cond_loss, marg_loss, reg_loss)
    # eval tuple = (kl, pr, sp)

    cdloss, mgloss, totloss, regloss, pcorrs, scorrs, kldivs = [], [], [], [], [], [], []

    with open(evalfile, 'r') as ef:
        for line in ef.readlines():
            val = line.split(',')
            kldivs.append(float(val[0]))
            pcorrs.append(float(val[1]))
            scorrs.append(float(val[2]))
    
    
    with open(lossfile, 'r') as lf:
        for line in lf.readlines():
            val = line.split(',')
            totloss.append(float(val[0]))
            cdloss.append(float(val[1]))
            mgloss.append(float(val[2]))
            regloss.append(float(val[3]))
    
    # cdloss = np.load(root + 'condloss.npy')
    # mgloss = np.load(root + 'margloss.npy')
    # pcorrs = np.load(root + 'pears_corrs.npy')
    # scorrs = np.load(root + 'spear_corrs.npy')
    # kldivs = np.load(root + 'kldivs.npy')

    
    plot_vec(cdloss, root + 'cond_loss.png', 'steps', 'loss', 'Conditional loss vs steps')
    plot_vec(mgloss, root + 'marg_loss.png', 'steps', 'loss', 'Marginal loss vs steps')
    plot_vec(totloss, root + 'total_loss.png', 'steps', 'loss', 'Total loss vs steps')
    plot_vec(regloss, root + 'reg_loss.png', 'steps', 'loss', 'Reg loss vs steps')
    
    plot_vec(pcorrs, root + 'pearcorrs.png', 'steps', 'correlation coef', 'Pearson Correlation vs steps')
    plot_vec(scorrs, root + 'spearcorrs.png', 'steps', 'correlation coef', 'Spearman Correlation vs steps')
    plot_vec(kldivs, root + 'kldivs.png', 'steps', 'KL value', 'Avg KL divergence vs steps')




# logfolder = '../log/'
logfolder = '/home/ninad/Desktop/Link-to-sem4/dsis/prob-emb/box-code/log/'
# for root, subdir, _ in os.walk(logfolder):
#     print(subdir)

for exp_dir in os.listdir(logfolder):
    exp_folder = logfolder + exp_dir + os.sep
    plot_all_in_logfolder(exp_folder)
    
