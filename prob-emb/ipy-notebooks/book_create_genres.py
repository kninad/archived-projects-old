import os 
import csv
from collections import defaultdict
import itertools
import ast
import numpy as np 
import pandas as pd
import pickle
import sys


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





def create_final_book_dict(csv_file, ratings_threshold, usernum_threshold):
    # using usecols in read_csv to exclude the last column of timestamp
    df = pd.read_csv(csv_file, delimiter=',', usecols=[0,1,2])    
    df = df[df['rating'] >= ratings_threshold]  # rating thresholding
    
    book_user_dict = defaultdict(list)
    for index, row in df.iterrows():    
        bookId = int(row['book_id'])
        usrId = int(row['user_id'])
        # rating = row['rating']       
        book_user_dict[bookId].append(usrId)

    # store the final res here, after the usernum thresholding
    final_dict = {}
    # num_users = 0
    for key,val in book_user_dict.items():
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
        book1, book2 = pair
        count_matrix[pair] = findnum_common_elements(final_dict[book1], final_dict[book2])
    
    # adding the marginal counts
    for k in final_dict.keys():
        pair = (k,k)
        count_matrix[pair] = len(final_dict[k])
    
    return count_matrix


def create_final_genres_dict(relevent_genfile, tags_csv, books_csv, book_tags_csv,
                             tag_count_threshold, final_book_ids):
        
    tmp_genres = []
    with open(relevent_genfile, 'r') as f:    
        for l in f.readlines():
            t = l.strip()            
            tmp_genres.append(t)

    tmp_genres[0] = '10th-century'
    good_genres = [ val for i,val in enumerate(tmp_genres) if i%2 == 0]

    df_tags = pd.read_csv(tags_csv, delimiter=',')
    df_tags.dataframeName = 'tags.csv'
    
    tag_id_list = df_tags['tag_id'].tolist()
    tag_name_list = df_tags['tag_name'].tolist()

    tag_id_to_name = {}
    tag_name_to_id = {}
    N = len(tag_id_list)

    for i in range(N):
        tagid = tag_id_list[i]
        tagnm = tag_name_list[i]
        
        if tagnm in good_genres:
            tag_id_to_name[tagid] = tagnm
            tag_name_to_id[tagnm] = tagid

    # converting from book-id to goodreads-id
    df_books = pd.read_csv(books_csv, delimiter=',')
    df_books.dataframeName = 'books.csv'
    df_books= df_books[['book_id', 'goodreads_book_id']]    

    #dict with key: goodreads_book_id, val: book_id 
    book_id_dict = defaultdict(lambda:-1)
    for index, row in df_books.iterrows():    
        bookId = int(row['book_id'])
        GoodReadId = int(row['goodreads_book_id'])
        book_id_dict[GoodReadId]=bookId

    #load book_tags file
    df_book_tags = pd.read_csv(book_tags_csv, delimiter=',')
    df_book_tags.dataframeName = 'book_tags.csv'    
    # tag_count_threshold = 500
    df_book_tags = df_book_tags[df_book_tags['count'] >= tag_count_threshold]
    
    #convert the good_read_book_id to book_id in this dataframe
    for index, row in df_book_tags.iterrows():
        bookId = book_id_dict[int(row['goodreads_book_id'])]
        df_book_tags.set_value(index,'goodreads_book_id', bookId)     

    # rename the column name from goodread_id to book_id
    df_book_tags.rename(columns={'goodreads_book_id': 'book_id'}, inplace=True)
    

    book_tagname_dict = defaultdict(list)
    for index, row in df_book_tags.iterrows():    
        bookId = int(row['book_id'])
        tagId = int(row['tag_id'])
        
        # Only conisder the tag if its in our good-tag list
        # Only consider the book if its in our final book dict.
        if (tagId in tag_id_to_name) and (bookId in final_book_ids):
            tagname = tag_id_to_name[tagId]
            book_tagname_dict[bookId].append(tagname)
    
    # GENRE COUNT MATRIX
    #marginal counts included
    #here the tag tag pair is incremented by one if a book is listed with both of the two tags
    tag_names_count =  defaultdict(int)
    for key, val in book_tagname_dict.items():
        for i in range(len(val)):
            for j in range(i, len(val)):
                # keeping a consistent ordering for the key
                key = (val[i], val[j]) if val[i] <= val[j] else (val[j], val[i])
                # name_key = (tag_id_to_name[key[0]], tag_id_to_name[key[1]])
                tag_names_count[key] += 1  

    return tag_id_to_name, tag_name_to_id, book_tagname_dict, tag_names_count

    


def create_vocab_marginal_files(book_count_matrix, book_num_users, genre_count_matrix, genre_num_users, datadir):
    
    # Create the marginals file and a separate vocab file (for all the possible movieIds)
    # Consider only the diagonal entries in count_matrix
    # Also, the 2 lists work as follows: 
    # marginals[i] == marginal_prob(movieid_vocab[i], count_matrix, num_users)
    # i.e the indices of both lists match on movieid and its marginal probability

    mov_marginals = []
    movieid_vocab = []
    for key_pair in book_count_matrix.keys():
        k1, k2 = key_pair
        if k1 == k2:        
            movieid_vocab.append(k1)
            mov_marginals.append(marginal_prob(k1, book_count_matrix, book_num_users))

    # Write out the lists to text files
    
    fname_mov_marginals = datadir + "book_marginal_prob.txt"
    with open(fname_mov_marginals, "w") as f:
        for prob in mov_marginals:
            f.write("%s\n" % prob)

    fname_mov_vocab = datadir + "book_vocabulary.txt"
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


def create_master_files(book_count_matrix, genre_count_matrix, genname_dict, datadir):
    REL = "IsA" 

    # BOOKS
    print("writing book master files")
    mov_tup_list = []
    for key_pair in book_count_matrix.keys():
        k1, k2 = key_pair
        if k1 != k2:
            prob_k2k1 = conditional_prob(k2, k1, book_count_matrix)
            tmp_tup1 = (REL, k1, k2, prob_k2k1)        
            mov_tup_list.append(tmp_tup1)
            
            prob_k1k2 = conditional_prob(k1, k2, book_count_matrix)
            tmp_tup2 = (REL, k2, k1, prob_k1k2)
            mov_tup_list.append(tmp_tup2)
    
    fname_mov_master = datadir + "book_book_master.txt"
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


    # BOOK - GENRES
    print("writing book-genre master files")
    tup_list = []
    for bookid in genname_dict:
        genres = genname_dict[bookid]
        for g in genres:
            tmp_tup = (REL, bookid, g, 1.0)
            tup_list.append(tmp_tup)
    
    fname_master = datadir + "book_genre_master.txt"
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
    
    # rawdata_file = rootdir + 'datasets/goodbooks-10k-master/samples/ratings.csv'
    rawdata_file = rootdir + 'datasets/goodbooks-10k-master/ratings.csv'
    
    datadir = rootdir + 'prob-emb/box-code/data/book_data/book_data_' + str(t_rating) + '_' + str(t_users) + '_taxonomy/'
    
    print(datadir)
    print(t_rating, t_users)


    final_dict = create_final_book_dict(rawdata_file, t_rating, t_users)
    book_num_users = get_total_users(final_dict)
    book_cmatrix = create_count_matrix(final_dict)
    final_bookids = list(final_dict.keys())
    print("Done with BOOK")
    del(final_dict)
    
    if not os.path.exists(datadir):
        os.makedirs(datadir)
    
    relevant_genfile = rootdir + 'datasets/goodbooks-10k-master/relevant-book-genres.txt'
    tags_csv = rootdir + 'datasets/goodbooks-10k-master/tags.csv'
    books_csv = rootdir + 'datasets/goodbooks-10k-master/books.csv'
    book_tags_csv = rootdir + 'datasets/goodbooks-10k-master/book_tags.csv'
    tag_count_threshold = 500
    
    tag_id_to_name, tag_name_to_id, book_tagname_dict, tag_names_count = create_final_genres_dict(relevant_genfile, 
                                            tags_csv, books_csv, book_tags_csv,
                                            tag_count_threshold, final_bookids)
        
    print("Writing MARG and VOCAB")
    create_vocab_marginal_files(book_cmatrix, book_num_users, tag_names_count, len(final_bookids), datadir)
    
    print("Writing master files")
    create_master_files(book_cmatrix, tag_names_count, book_tagname_dict, datadir)


if __name__ == "__main__":
    main()
