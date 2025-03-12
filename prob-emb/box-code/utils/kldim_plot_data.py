import numpy as np 
import os
import sys
from matplotlib import pyplot as plt

def update_dict_for_logfolder(experiment_folder, glob_resdict):
    root = experiment_folder
    
    dev_resfile = root + 'dev_results.txt'
    with open(dev_resfile, 'r') as rfile:
        num_res = rfile.readline()
    
    index = root.find('dim')
    dim = root[index+3:index+5]
    if dim == '5_':
        key = 5
    else:
        key = int(dim)

    val = str(num_res)
    vlist = val.split(',')
    glob_resdict[key] = vlist   

    return

code_root = '/home/ninad/Desktop/Link-to-sem4/dsis/ninad_gypsum/'
#logfolder = code_root + 'log/' 
logfolder = code_root + 'holding/' 

glob_dict = {}
for exp_dir in os.listdir(logfolder):
    exp_folder = logfolder + exp_dir + os.sep    
    update_dict_for_logfolder(exp_folder, glob_dict)

dims = []
kldiv = []
pcorr = []
scorr = []
for key in sorted(glob_dict):
    val = glob_dict[key]
    dims.append(key)
    kldiv.append(float(val[0]))
    pcorr.append(float(val[1]))
    scorr.append(float(val[2]))

#print dims
#print kldiv
#print pcorr
#print scorr


def plot_vec(xvec, yvec, fname, xlab, ylab, title):
    plt.figure()
    plt.plot(xvec, yvec, 'g-')
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title)
    plt.savefig(fname)
    plt.close()
    return

def plot_corr(xvec, pcor, scor, fname, xlab, ylab, title):
    plt.figure()
    plt.plot(xvec, pcor, 'b-', label='Pearson Corr')
    plt.plot(xvec, scor, 'r-', label='Spearman Corr')
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title)
    plt.legend(loc='lower right')
    plt.savefig(fname)
    plt.close()
    return

plot_corr(dims, pcorr, scorr, './cor_dim.png', 'Dimension', 'Correlations', 'Eval Correlations vs Embedding dimension')

plot_vec(dims, kldiv, './kld_dim.png', 'Dimension', 'KL', 'Eval KL divergence vs Embedding dimension')








