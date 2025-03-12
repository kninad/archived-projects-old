import os 
import csv
from collections import defaultdict
import itertools
import ast
import numpy as np 
import pandas as pd
import pickle
import sys


def create_final_mov_dict(csv_file, ratings_threshold, usernum_threshold):
    # using usecols in read_csv to exclude the last column of timestamp
    df = pd.read_csv(csv_file, delimiter=',', usecols=[0,1,2])    
    df = df[df['rating'] >= ratings_threshold]  # rating thresholding
    
    movie_user_dict = defaultdict(list)
    for index, row in df.iterrows():    
        movId = int(row['movieId'])
        usrId = int(row['userId'])
        # rating = row['rating']       
        movie_user_dict[movId].append(usrId)

    # store the final res here, after the usernum thresholding
    final_dict = {}
    # num_users = 0
    for key,val in movie_user_dict.items():
        if len(val) >= usernum_threshold:
            final_dict[key] = val
    
    return final_dict

def get_total_users(final_dict):
    total_users = set()
    for mov in final_dict.keys():
        row = set(final_dict[mov])
        total_users = total_users.union(row)
    num_users = len(total_users)
    return num_users

def create_count_matrix(final_dict):
    # helper function
    def findnum_common_elements(list1, list2):
        return len(set(list1).intersection(list2))

    count_matrix = {}
    for pair in itertools.combinations(final_dict.keys(), r=2):
        mov1, mov2 = pair
        count_matrix[pair] = findnum_common_elements(final_dict[mov1], final_dict[mov2])
    
    # adding the marginal counts
    for k in final_dict.keys():
        pair = (k,k)
        count_matrix[pair] = len(final_dict[k])
    
    return count_matrix


def get_genres(genres_json, id_to_genre):
    '''
    id_to_genre: its like a global dict to store the mappings 
                from the genre-id to genre-name
    '''
    if not genres_json:
        return None
    
    tmp = ast.literal_eval(genres_json)
    ids_list = []
    gen_list = []           
    
    for d in tmp:
        id_num = d['id']
        genre_name = d['name']
        genre_name = genre_name.replace(" ", "-") # change all whitespaces to hyphen
        ids_list.append(id_num)
        gen_list.append(genre_name)
        
        if id_num not in id_to_genre:
            id_to_genre[id_num] = genre_name
    
    return ids_list, gen_list, id_to_genre


def create_final_genres_dict(csv_file, movIdsList):    
    df_meta = pd.read_csv(csv_file, delimiter=',')    
    df_meta = df_meta[['genres', 'id']]
    
    genid_dict = {}
    genname_dict = {}
    genid_counts = defaultdict(int) # counting the genre-genre pairs
    id_to_genre = {}

    for index, row in df_meta.iterrows():    
        
        try:
            movId = int(row['id'])
            gen_json = row['genres']

            if movId in movIdsList:  # Only consider the thresholded movies
                ids, genres, id_to_genre = get_genres(gen_json, id_to_genre)    
                genid_dict[movId] = ids
                genname_dict[movId] = genres

                for i in ids:
                    singleton_pair = (i, i)
                    genid_counts[singleton_pair] += 1

                for pair in itertools.combinations(ids, r=2):
                    id1, id2 = pair
                    genid_counts[pair] += 1
                
        except Exception as e:
            print(index, row['id'])
            repr(e)
    
    return genid_dict, genname_dict, genid_counts, id_to_genre


def marginal_prob(obj_id, count_matrix, num_users):
    '''function to get the marginal prob:
        P(movie_id1)       
    '''
    margn_count = count_matrix[(obj_id, obj_id)]
    return margn_count * 1.0/num_users

def joint_prob(obj_id1, obj_id2, count_matrix, num_users):
    '''function to get the joint prob:
        P(movie_id1, movie_id2)       
    '''
    if (obj_id1, obj_id2) not in count_matrix.keys():
        key = (obj_id2, obj_id1)
    else:
        key = (obj_id1, obj_id2)
    
    joint_count = count_matrix[key]    
    return joint_count * 1.0/num_users

def conditional_prob(obj_id1, obj_id2, count_matrix):
    '''function to get the conditional prob:
        P(movie_id1 | movie_id2)       
    '''
    if (obj_id1, obj_id2) not in count_matrix.keys():
        key = (obj_id2, obj_id1)
    else:
        key = (obj_id1, obj_id2)
    
    joint_count = count_matrix[key]
    margn_count = count_matrix[obj_id2, obj_id2]
    
    return joint_count * 1.0/margn_count


def create_vocab_marginal_files(movie_count_matrix, movie_num_users, genre_count_matrix, genre_num_users, datadir):
    
    # Create the marginals file and a separate vocab file (for all the possible movieIds)
    # Consider only the diagonal entries in count_matrix
    # Also, the 2 lists work as follows: 
    # marginals[i] == marginal_prob(movieid_vocab[i], count_matrix, num_users)
    # i.e the indices of both lists match on movieid and its marginal probability

    mov_marginals = []
    movieid_vocab = []
    for key_pair in movie_count_matrix.keys():
        k1, k2 = key_pair
        if k1 == k2:        
            movieid_vocab.append(k1)
            mov_marginals.append(marginal_prob(k1, movie_count_matrix, movie_num_users))

    # Write out the lists to text files
    
    fname_mov_marginals = datadir + "movie_marginal_prob.txt"
    with open(fname_mov_marginals, "w") as f:
        for prob in mov_marginals:
            f.write("%s\n" % prob)

    fname_mov_vocab = datadir + "movie_vocabulary.txt"
    with open(fname_mov_vocab, "w") as f:
        for movid in movieid_vocab:
            f.write("%s\n" % movid)

    gen_marginals = []
    gen_vocab = []
    for key_pair in genre_count_matrix.keys():
        k1, k2 = key_pair
        if k1 == k2:        
            gen_vocab.append(k1)
            gen_marginals.append(marginal_prob(k1, genre_count_matrix, genre_num_users))

    # Write out the lists to text files
    fname_gen_marginals = datadir + "genre_marginal_prob.txt"
    with open(fname_gen_marginals, "w") as f:
        for prob in gen_marginals:
            f.write("%s\n" % prob)

    fname_gen_vocab = datadir + "genre_vocabulary.txt"
    with open(fname_gen_vocab, "w") as f:
        for gen in gen_vocab:
            f.write("%s\n" % gen)
    return 


def create_master_files(movie_count_matrix, genre_count_matrix, genname_dict, datadir):
    REL = "IsA" 

    # MOVIES
    print("writing movie master files")
    mov_tup_list = []
    for key_pair in movie_count_matrix.keys():
        k1, k2 = key_pair
        if k1 != k2:
            prob_k2k1 = conditional_prob(k2, k1, movie_count_matrix)
            tmp_tup1 = (REL, k1, k2, prob_k2k1)        
            mov_tup_list.append(tmp_tup1)
            
            prob_k1k2 = conditional_prob(k1, k2, movie_count_matrix)
            tmp_tup2 = (REL, k2, k1, prob_k1k2)
            mov_tup_list.append(tmp_tup2)
    
    fname_mov_master = datadir + "movie_movie_master.txt"
    with open(fname_mov_master, "w") as f:
        writer = csv.writer(f, delimiter='\t', lineterminator='\n')
        writer.writerows(mov_tup_list)


    # GENRES
    print("writing genre master files")
    gen_tup_list = []
    for key_pair in genre_count_matrix.keys():
        k1, k2 = key_pair
        if k1 != k2:
            prob_k2k1 = conditional_prob(k2, k1, genre_count_matrix)
            tmp_tup1 = (REL, k1, k2, prob_k2k1)        
            gen_tup_list.append(tmp_tup1)
            
            prob_k1k2 = conditional_prob(k1, k2, genre_count_matrix)
            tmp_tup2 = (REL, k2, k1, prob_k1k2)
            gen_tup_list.append(tmp_tup2)
    
    fname_master = datadir + "genre_genre_master.txt"
    with open(fname_master, "w") as f:
        writer = csv.writer(f, delimiter='\t', lineterminator='\n')
        writer.writerows(gen_tup_list)


    # MOVIE - GENRES
    print("writing movie-genre master files")
    tup_list = []
    for movid in genname_dict:
        genres = genname_dict[movid]
        for g in genres:
            tmp_tup = (REL, movid, g, 1.0)
            tup_list.append(tmp_tup)
    
    fname_master = datadir + "movie_genre_master.txt"
    with open(fname_master, "w") as f:
        writer = csv.writer(f, delimiter='\t', lineterminator='\n')
        writer.writerows(tup_list)


    return



def main():
    # t_rating = 4
    # t_users = 100
    t_rating = float(sys.argv[1])
    t_users = int(sys.argv[2])        

    # rootdir = '/home/nkhargonkar/dsis/'
    rootdir = '/home/ninad/Desktop/Link-to-sem4/dsis/'    
  
    # rawdata_file = rootdir + 'datasets/the-movies-dataset/ratings_small.csv'  # SMALL RATINGS FILE
    rawdata_file = rootdir + 'datasets/the-movies-dataset/ratings.csv'    # FULL RATINGS FILE

    metdata_file = rootdir + 'datasets/the-movies-dataset/movies_metadata.csv'  # for the genres
    datadir = rootdir + 'prob-emb/box-code/data/movie_data/movie_data_' + str(t_rating) + '_' + str(t_users) + '_taxonomy/'

    print(datadir)
    print(t_rating, t_users)

    final_dict = create_final_mov_dict(rawdata_file, t_rating, t_users)
    mov_num_users = get_total_users(final_dict)
    mov_cmatrix = create_count_matrix(final_dict)
    movId_list = list(final_dict.keys())
    print("Done with MOVIE")
    del(final_dict)
    
    if not os.path.exists(datadir):
        os.makedirs(datadir)

    
    genid_dict, genname_dict, genid_counts, id_to_genre = create_final_genres_dict(metdata_file, movId_list)
    genname_counts = {}   # the count matrix for genres with their names 
    for k,v in genid_counts.items():
        k1, k2 = k
        g1 = id_to_genre[k1]
        g2 = id_to_genre[k2]
        genname_counts[g1, g2] = v
    print("Done with GENRES")

    print("Writing MARG and VOCAB")
    create_vocab_marginal_files(mov_cmatrix, mov_num_users, genname_counts, len(movId_list), datadir)
    
    print("Writing master files")
    create_master_files(mov_cmatrix, genname_counts, genname_dict, datadir)
    
    
    # trn_file = datadir + 'movie_train.txt'
    # trn_eval_file = datadir + 'movie_train_eval.txt'
    # create_trneval_file(trn_file, trn_eval_file, 10)


if __name__ == "__main__":
    main()
