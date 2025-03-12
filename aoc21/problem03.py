'''
Problem 02
==========

Reading in from the text file

Loading the numbers in the file in an array

Algorithm:

Brute Force approach -- for K-bit number, simply keep track of counts of 1 and 0 for
that bit position.
'''
import numpy as np
INPUT_FILE_PATH = "input/prob3.input.txt"

def parse_input(filepath):
    with open(filepath) as f:
        lines = f.readlines()
    
    lines = [l.strip() for l in lines]
    return lines

def convert_to_np(str_array):
    N = len(str_array)
    bit_length = len(str_array[0])
    data = np.zeros((N, bit_length))
    for i, bitmask in enumerate(str_array):
        # for j, c in enumerate(bitmask):
        #     if c == '1':
        #         data[i][j] = 1
        data[i] = np.array([int(c) for c in bitmask])
    return data

def get_counts(data) -> None:
    N = len(data)
    one_counts = np.sum(data, axis=0)
    zero_counts =  N - one_counts
    return one_counts, zero_counts

def get_dominant_bits(array):
    bit_length = len(array[0])
    c_one, c_zero =  get_counts(array)
    dominant = np.ones(bit_length)
    dominant[c_zero > c_one] = 0
    return dominant

def oxygen_rating(array, dominant):
    pass

def co2_rating(array, dominant):
    pass




def main():
    print("Answer: ")


if __name__ == "__main__":
    main()