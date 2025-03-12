'''
Problem 01
==========

Reading in from the text file

Loading the numbers in the file in an array

Doing the comparisons on the consecutive values 
'''

INPUT_FILE = "input/prob1.input.txt"


def format_lines(lines):
    new_format = [''] * len(lines)
    for idx, l in enumerate(lines):
        new_format[idx] = int(l.strip())
    return new_format


def total_increasing(array: "list[int]") -> int:
    count = 0
    for i in range(1, len(array)):
        if array[i] > array[i-1]:
            count += 1
    return count


def total_increasing_sliding_window(array: "list[int]", window_size: int) -> int:
    count = 0
    # window_size = 3
    prev_runsum = 0
    curr_sum = 0
    for i in range(window_size):
        prev_runsum += array[i]

    for i in range(window_size, len(array)):
        curr_sum = prev_runsum + array[i] - array[i-window_size]
        if curr_sum > prev_runsum:
            count += 1
        prev_runsum = curr_sum
    return count


def main():
    with open(INPUT_FILE) as f:
        lines = f.readlines()

    lines = format_lines(lines)
    count = total_increasing(lines)
    sw_count = total_increasing_sliding_window(lines, 3)

    print("Part 1: ", count)
    print("Part 2: ", sw_count)


if __name__ == "__main__":
    main()
