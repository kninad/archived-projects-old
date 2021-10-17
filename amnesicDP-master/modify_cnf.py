
from pprint import pprint
from collections import defaultdict
import copy

import numpy as np


Eps = "e"   # empty symbol
# assumption start symbol is "S"
# cnfFile = "./cnf_balance.txt"

def cnf_input(cnfFile):
    score = defaultdict(int)
    rules = defaultdict(list)
    allparents = defaultdict(list)

    with open(cnfFile, 'r') as rfile:
        lines = rfile.read()

    for line in lines.strip().split("\n"):
        if not line.strip(): 
            continue
        
        parent, children = line.split(" -> ")
        children = children.split()
        rules[parent].extend(children)    # a list
        
        for child in children:      # reverse lookup hashmap
            allparents[child].append(parent)
            score[(parent, child)] = 0  # score for original rules is 0

    return (rules, allparents, score)


def find_terms(rules):
    terms = set()
    non_terms = set(rules.keys())   # the keys will be the non-terminals

    for _,val in rules.items():
        if len(val) == 1 and len(val[0]) == 1:  # its a terminal symbol
            terms.add(val[0])

    return (terms, non_terms)
    

def modify_grammar(rules, score, terms, non_terms):

    old_rules = copy.deepcopy(rules)

    insKey = "I"    # insertion symbol
    rules[insKey].append(Eps)
    score[insKey, Eps] = 0
    
    for i in terms:
        nkey = "X_" + i
        rules[nkey].append(i)
        score[(nkey, i)] = 0

        # Substitution
        for j in terms:
            if j != i:
                subsKey = "X_" + i + "_" + j
                rules[subsKey].append(j)
                score[(subsKey, j)] = 1
                

        # Insertion        
        insert_list = [nkey+insKey, insKey+nkey]
        for item in insert_list:
            rules[insKey].append(item)
            score[insKey, item] = 1


    for N in non_terms: # non_terms not updated yet
        rules[N].append(N + insKey)
        rules[N].append(insKey + N)
        score[N, N + insKey] = 0
        score[N, insKey + N] = 0
    
    # Deletion
    for k,v in old_rules.items():
        if(len(v)==1 and len(v[0]) == 1):
            rules[k].append(Eps)
            score[k, Eps] = 1
    
    rules["S"].append(Eps)
    score["S", Eps] = 0

    non_terms = set(rules.keys())

    return (rules, score, terms, non_terms)


# Function to check whether variable is nullable # also returns the score
def find_null(rules, score, terms, non_terms):
    null_score = defaultdict(int)
    null_set = set()

    for key, vals in rules.items():
        if Eps in vals:
            null_set.add(key)
            null_score[key] = score[key, Eps]

    flag = 1
    while(flag):
        flag = 0
        for key, vals in rules.items():
            if key not in null_set:
                tmpScore = np.inf   # infinity
                for v in vals:
                    chk = True
                    vScore = 0
                    for i in v:
                        if i not in null_set:
                            chk = False
                            break
                        else:
                            vScore += null_score[i]
                            # or should vScore = score[key, v] ???
                    if chk:
                        null_set.add(key)
                        tmpScore = min(tmpScore, vScore)
                        flag = 1    # Nullable variable found!                                        
                null_score[key] = tmpScore

    return (null_set, null_score)


def create_delete(rules, score, non_terms, old_non_terms, null_score, str_len):
    
    null_list = [(k,v) for k,v in null_score.items() 
                    if (k in old_non_terms) and (v <= str_len)]
    
    del_dict = {}    
    
    # for k in old_non_terms:
    for k in non_terms:
        # print(k)        
        del_dict[k] = [(k,0)]   # should ideally be a set or a hashmap
        
        count = True 
        while(count):
            count = False
            for (x,a) in del_dict[k]:
                for (y,b) in null_list:
                    for rkey, rval in rules.items():
                        found = 0
                        if (x+y in rval) and (score[rkey, x+y] + a + b <= str_len):
                            tscore = score[rkey, x+y] + a + b
                            found = 1
                        elif (y+x in rval) and (score[rkey, y+x] + a + b <= str_len):
                            tscore = score[rkey, y+x] + a + b
                            found = 1
                        
                        if found == 1:
                            temp_flag = 0
                            for delk_item in del_dict[k]:
                                if rkey == delk_item[0]:
                                    temp_flag = 1
                                    if tscore < delk_item[1]:
                                        del_dict[k].remove(delk_item)
                                        del_dict[k].append((rkey, tscore))
                                        count = True   # updated dict
                                    break

                            if temp_flag == 0:
                                del_dict[k].append((rkey, tscore))                        
                                count = True   #updated dict

    return (del_dict, null_list)



def standard_dp(string, rules, score, del_dict):
    # create the NxN dp table
    str_len = len(string)

    # T = [ [[] for i in range(str_len)] for j in range(str_len)]     # should ideally be a list of sets
    T = [ [ {} for i in range(str_len)] for j in range(str_len)]     # should ideally be a list of sets
    # why not even a hashmap ??

    # init steps
    for i, a in enumerate(string):
        for key, vals in rules.items():
            if a in vals:
                s = score[(key, a)]
                if key in T[i][i]:
                    # if T[i][i][key] >= s:
                    #     T[i][i][key] = s 
                    T[i][i][key] = min(T[i][i][key], s)
                else:
                    T[i][i][key] = s

                for (y, t) in del_dict[key]:
                    tmp_score = t+s
                    if y in T[i][i]:
                        T[i][i][y] = min(T[i][i][y], tmp_score)
                    else:
                        T[i][i][y] = tmp_score

    # recurrence
    for t in range(2, str_len+1):   # t is length of substring considered
        num_steps = str_len - t + 1
        for i in range(num_steps):   # i is the 1st index in T
            new_id = i + t - 1  # j+1
            for l in range(i,new_id):   
                d1 = T[i][l]
                d2 = T[l+1][new_id]

                for k1,v1 in d1.items():
                    for k2,v2 in d2.items():
                        for z,vals in rules.items():
                            if k1+k2 in vals:
                                s = score[(z, k1+k2)]
                                if z in T[i][new_id]:
                                    T[i][new_id][z] = min(T[i][new_id][z], 
                                                            s+v1+v2) 
                                else:
                                    T[i][new_id][z] = s + v1 + v2
            old_dict = copy.deepcopy(T[i][new_id])
            for xkey, xscore in old_dict.items():
                for ykey, yscore in del_dict[xkey]:
                    b = yscore + xscore
                    if b <= str_len:
                        if ykey in T[i][new_id]:
                            T[i][new_id][ykey] = min(T[i][new_id][ykey], yscore)
                        else:
                            T[i][new_id][ykey] = yscore

                

    return T    # T(1,n)



def main():   # test function
    cfile = "./cnf_balance.txt"
    string = "aaa"
    str_len = len(string)

    rules, allparents, score = cnf_input(cfile)
    # pprint(rules)
    # pprint(score)    
    terms, non_terms = find_terms(rules)
    # pprint(terms)
    # pprint(non_terms)
    
    old_non_terms = copy.deepcopy(non_terms)
    rules, score, terms, non_terms = modify_grammar(rules, score, terms, non_terms)    
    # pprint(rules)
    # pprint(score)
    # pprint(non_terms)
    ### correc upto this point


    null_set, null_score = find_null(rules, score, terms, non_terms)
    # pprint(null_set)
    # pprint(null_score)
    deld, nlist = create_delete(rules, score, non_terms, old_non_terms, null_score, str_len)
    # pprint(deld)
    # pprint(nlist)

    final = standard_dp(string, rules, score, deld)
    pprint(final[0][str_len - 1])


if __name__ == "__main__":
    main()





