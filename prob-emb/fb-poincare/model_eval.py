
from hype.graph import eval_reconstruction, load_adjacency_matrix
import argparse
import numpy as np
import torch
import os
from hype.lorentz import LorentzManifold
from hype.euclidean import EuclideanManifold, TranseManifold
from hype.poincare import PoincareManifold
import pickle

def get_model_info(fname):
    chkpnt = torch.load(fname)
    dset = chkpnt['conf']['dset']
    if not os.path.exists(dset):
        raise ValueError("Can't find dset!")
    data_format = 'hdf5' if dset.endswith('.h5') else 'csv'
    # dset = load_adjacency_matrix(dset, 'hdf5')
    dset = load_adjacency_matrix(dset, data_format)
    
    info_dict = {}    
    # info_dict['dset'] = dset
    info_dict['idmap'] = dset['idmap']
    info_dict['embeddings'] = chkpnt['embeddings'].numpy()
    return info_dict

def get_vector(entity_id, info_dict):
    imap = info_dict['idmap']
    idx_obj = imap[entity_id]
    vec = info_dict['embeddings'][idx_obj, :]
    return vec

def distance(u, v, eps=1e-5):
    # Calculate the distance between two vectors in poincare space
    squnorm = np.clip(np.sum(u * u, axis=-1), 0, 1 - eps)
    sqvnorm = np.clip(np.sum(v * v, axis=-1), 0, 1 - eps)
    sqdist = np.sum(np.power(u - v, 2), axis=-1)
    x = 2 * sqdist / ((1 - squnorm) * (1 - sqvnorm)) + 1
    # arcosh calc
    tmp = np.sqrt(np.power(x, 2) - 1)
    return np.log(x + tmp)

# just an added function
def dist_norms(u, v):
    norm_u = np.linalg.norm(u, 2)
    norm_v = np.linalg.norm(v, 2)
    return (norm_u - norm_v)

def score_func_isa(u, v, alpha=1000):
    norm_u = np.linalg.norm(u, 2)
    norm_v = np.linalg.norm(v, 2)    
    tmp = 1 + alpha * (norm_v - norm_u)
    return -1 * tmp * distance(u, v)
    
def score_func_rank(u, v, beta=0.1):
    norm_u = np.linalg.norm(u, 2)
    norm_v = np.linalg.norm(v, 2)    
    tmp = 1 + beta * np.abs(norm_v - norm_u)
    return tmp * distance(u, v)

def compute_nn(ent_id, other_ids, info_dict, func='dist', alpha=1000, beta=0.1):
    vec = get_vector(ent_id, info_dict)
    dists = []
    for ent in other_ids:
        vec_ot = get_vector(ent, info_dict)
        if func == 'dist':
            tmp_score = distance(vec, vec_ot)
        elif func == 'isa':
            tmp_score = score_func_isa(vec, vec_ot, alpha)
        elif func == 'rank':
            tmp_score = score_func_rank(vec, vec_ot, beta)
        dists.append(tmp_score)
    
    dists = np.asarray(dists)
    sorted_ids = np.argsort(dists)    
    return dists, sorted_ids

def sort_by_norm(info_dict):
    emb_mat = info_dict['embeddings']
    imap = info_dict['idmap']
    norms = np.linalg.norm(emb_mat, axis=1)
    sorted_ids = np.argsort(norms)
#    sorted_ids = list(sorted_ids)
    # smaller norms will be at the top of the heirarchy
    
    rev_imap = {}   
    for key, val in imap.items():
        rev_imap[val] = key
    
    sorted_names = []
    for i in range(len(sorted_ids)):
        init_val = sorted_ids[i]
        new_val = rev_imap[init_val]
        # update with the entity_id
        sorted_names.append(new_val)
     
    return sorted_ids, sorted_names, norms
        

#def main():    
#    np.random.seed(42)

#    parser = argparse.ArgumentParser(description='Process the model chkpnt path')
#    parser.add_argument('file', help='Path to checkpoint')
#    args = parser.parse_args()
#    
#    model_file = args.file
#    info_dict = get_model_info(model_file)
#    

#if __name__ == '__main__':
#    main()

