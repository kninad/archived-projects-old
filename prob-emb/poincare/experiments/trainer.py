"""
Main Training file for the Gensim Poincare Model
From the supplied arguments, loads a dataset, trains
and then saves the logs, params and args to appropriate folders
"""
import os
import sys
import logging
import argparse
from datetime import datetime
import time
from argparse import ArgumentParser
sys.path.insert(0, os.getcwd())

import pickle
import numpy as np
import pandas as pd
from gensim.models.poincare import PoincareModel, PoincareKeyedVectors, PoincareRelations

from utils import data_loader


def train_run(args):
    # create experiment name from args
    # create log folder, params folder from exp name
    # Start logging in exp log folder
    # save trained model in exp params folder
    
    exp_name = 'HB'+'time' + str(datetime.now()) + '_EXP' + str(args.train_dir) + \
    '_prbt' + str(args.prob_threshold) + '_reg' + str(args.reg_coef) + \
    '_dim' + str(args.embed_dim) + '_lr' + str(args.learning_rate) + \
    '_neg' + str(args.negs) + '_epoc' + str(args.epochs) + '_burnin' + str(args.burn_in)

    exp_name = exp_name.replace(":", "-")
    exp_name = exp_name.replace("/", "-")
    exp_name = exp_name.replace(" ", "-")        
    print(exp_name)
    
    # Training Logs Folder
    exp_log_folder = args.log_folder + exp_name + '/'              
    if not os.path.exists(exp_log_folder):
        os.makedirs(exp_log_folder)
        
    logging_file = exp_log_folder + 'logging.txt'
    logging.basicConfig(filename=logging_file, level=logging.INFO)
    
    # Model saving folder
    exp_params_folder = args.params_folder + exp_name + '/'
    if not os.path.exists(exp_params_folder):
        os.makedirs(exp_params_folder)
    
    training_file = args.train_dir + args.trn_file
    trn_dataset = data_loader.get_data_list(training_file, args.prob_threshold)
    print("Number of training examples: ", len(trn_dataset))
    
    # Create the model definition
    model = PoincareModel(train_data=trn_dataset, size=args.embed_dim, alpha=args.learning_rate,
                    negative=args.negs, regularization_coeff=args.reg_coef, 
                    burn_in=args.burn_in, burn_in_alpha=args.burn_in_alpha, 
                    init_range=args.init_range, seed=args.random_seed)

    # Start the model training
    model.train(epochs=args.epochs, batch_size=args.batch_size, print_every=args.print_every)
    
    # Save the model
    model_save_name = exp_params_folder + 'gensim_model.params'
    model.save(model_save_name)    
    
    # Save the arguments in the params folder
    args_fname = exp_params_folder + 'args_model.pkl'
    with open(args_fname, "wb") as f:
        pickle.dump(args, f)
    
    return

def main():
    # Basics
    parser = argparse.ArgumentParser(description='Process the model parameters')
    parser.add_argument('--random_seed', type=int, default=20180112, help='Random seed for model reproducibility')
    parser.add_argument('--save', type=bool, default=True, help='Save the model')
    parser.add_argument('--log_folder', type=str, default='./log/', help='Log folder location')
    parser.add_argument('--params_folder', type=str, default='./params/', help='Params folder location')
    
    # Dataset 
    parser.add_argument('--train_dir', type=str, default='./data/book_data/exp2.3_baseline_notaxo/', help='Directory for data files')
    
    parser.add_argument('--trn_file', type=str, default='book_train.txt', help='Training file')
    parser.add_argument('--trn_eval', type=str, default='book_train_eval.txt', help='Trn eval file')
    parser.add_argument('--dev_file', type=str, default='book_dev.txt', help='Dev data file')
    parser.add_argument('--tst_file', type=str, default='book_test.txt', help='Test data file')
    parser.add_argument('--marg_prob_file', type=str, default='book_marginal_prob.txt', help='which marginal probability file to use')

    parser.add_argument('--prob_threshold', type=float, default=0.5, help='Conditional probability threshold')
    
    # Model define params
    parser.add_argument('--embed_dim', type=int, default=50, help='Number of dimensions of the trained model')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning rate for training (alpha)')
    parser.add_argument('--negs', type=int, default=10, help='Number of negative samples to use')
    parser.add_argument('--reg_coef', type=float, default=1.0, help='Coefficient used for l2-regularization while training')
    parser.add_argument('--burn_in', type=int, default=10, help='Number of epochs to use for burn-in initialization')
    parser.add_argument('--burn_in_alpha', type=float, default=0.01, help='Learning rate for burn-in initialization')
    parser.add_argument('--init_range', type=tuple, default=(-0.001, 0.001), help='2-tuple Range within which the vectors are randomly initialized')
    
    # Model trainig
    parser.add_argument('--epochs', type=int, default=1, help='Number of iterations (epochs) over the corpus')
    parser.add_argument('--batch_size', type=int, default=20, help='Number of examples to train on in a single batch')
    parser.add_argument('--print_every', type=int, default=500, help='Prints progress and average loss after every `print_every` batches')
    
    # Parse the arguments                              
    args = parser.parse_args()
    
    # Run the trianing function with the args
    train_run(args)
    return

if __name__ == '__main__':
    main()
