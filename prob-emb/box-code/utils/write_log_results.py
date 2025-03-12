import numpy as np 
import os
import sys

def add_all_in_logfolder(experiment_folder, global_reslist):
    root = experiment_folder
    
    res_list = ''
    dev_resfile = root + 'dev_results.txt'
    with open(dev_resfile, 'r') as rfile:
        num_res = rfile.readline()
    
    index = root.find('_EXP.-')
    exp_name = root[index+6:]
    res_list += exp_name
    res_list += '\n'
    res_list += str(num_res)
    res_list += '\n'
#    print(res_list)

    global_reslist.append(str(res_list))
    return


def dict_for_logfolder(experiment_folder, glob_resdict):
    root = experiment_folder
    
    dev_resfile = root + 'dev_results.txt'
    with open(dev_resfile, 'r') as rfile:
        num_res = rfile.readline()
    
    index = root.find('_EXP.-')
    exp_name = root[index+6:]
    
    key = exp_name
    val = str(num_res)
    glob_resdict[key] = val
    

    return

code_root = '/home/ninad/Desktop/Link-to-sem4/dsis/ninad_gypsum/'
#logfolder = code_root + 'log/' 
logfolder = code_root + 'holding/' 

glob_dict = {}
for exp_dir in os.listdir(logfolder):
    exp_folder = logfolder + exp_dir + os.sep    
    dict_for_logfolder(exp_folder, glob_dict)

dict_file = code_root + 'final_results_comp_run3.txt'

with open(dict_file, "w") as wfile:
    wfile.write("Results:\n\n")
    for key in sorted(glob_dict):
        wfile.write(key)
        wfile.write('\n')
        wfile.write(glob_dict[key])
        wfile.write('\n\n')
