'''
Problem 02
==========

Reading in from the text file

Loading the numbers in the file in an array

Algorithm --> Simple parsing and if-else statements
'''

INPUT_FILE_PATH = "input/prob2.input.txt"


def parse_input(filepath):
    with open(filepath) as f:
        lines = f.readlines()

    depth = 0
    horzp = 0
    aim = 0

    for line in lines:
        command, value = line.split()

        if command == 'forward':
            horzp += int(value)
            depth += aim * int(value)
        elif command == 'up':
            aim -= int(value)
        elif command == 'down':
            aim += int(value)

    print("Horizontal Position: ", horzp)
    print("Depth: ", depth)
    print("Aim: ", aim)

    return depth * horzp


def main():
    print("Answer: ", parse_input(INPUT_FILE_PATH))


if __name__ == "__main__":
    main()
