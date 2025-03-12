
import pandas as pd
import numpy as np

def get_data_list(data_filepath, prob_threshold):
    """Function to create a list of training tuples. It is according
    to the training data format specified for the poincare model.
    
    Parameters
    ----------
    data_filepath : str
        csv file Path having the conditional probabilities. format is
        like - IsA \t term1 \t term2 \t prob.    
    prob_threshold : float
        threshold for the conditional probability. Only pairs having 
        prob greater than this will be considered.
    outFile : str
        File name to which output the modified training file
    
    Returns
    -------
    data_list : list
        List of training tuples (pairs) having 2 terms which satisfy 
        the threshold requirement.             
    """
    # data_list = []    
    df = pd.read_csv(data_filepath, header=None, delimiter='\t', usecols=[1,2,3])
    df.columns = ['t1', 't2', 'cond_prob']
    df = df[df.cond_prob >= prob_threshold]
    # drop the 3rd column now., since no use
    df.drop('cond_prob', axis=1, inplace=True)
    data_list = list(df.itertuples(index=False, name=None))

    outFile = data_filepath + "_HB_gensim.csv"
    with open(outFile, "w") as out:
        out.write("id1,id2,weight\n")
        for row in data_list:
            out.write("%s,%s,1\n" % (row[0], row[1]))
    return data_list
    

def write_mod_data(data_filepath, prob_threshold=0.01):
    """Function to create a list of training tuples. It is according
    to the training data format specified for the poincare model.
    
    Parameters
    ----------
    data_filepath : str
        csv file Path having the conditional probabilities. format is
        like - IsA \t term1 \t term2 \t prob.    
    prob_threshold : float
        threshold for the conditional probability. Only pairs having 
        prob greater than this will be considered.

    Returns
    -------
    data_list : list
        List of training tuples (pairs) having 2 terms which satisfy 
        the threshold requirement.             
    """
    # data_list = []    
    df = pd.read_csv(data_filepath, header=None, delimiter='\t', usecols=[1,2,3])
    df.columns = ['id1', 'id2', 'weight']
    df = df[df.weight >= prob_threshold]
    index = data_filepath.find('.txt')
    outFile = data_filepath[:index] + "_hb.csv"
    df.to_csv(outFile, sep=',', index=False)
    
    return None


