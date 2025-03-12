
import pandas as pd
import numpy as np

def get_data_list(data_filepath, prob_threshold, outFile="output.txt"):
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
    with open(outFile, "w") as out:
        for row in data_list:
            out.write("%d,%d\n" % (row[0], row[1]))
    return data_list
    



