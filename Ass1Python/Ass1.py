#Taha Koleilat 40263451 Muhammad Sarim 40261752

import numpy as np

def count_row_mismatches(first, second):
    number_of_mismatches = 0

    for i in range(len(first)):
        if first[i][2] != second[i][0]:
            number_of_mismatches += 1

    return number_of_mismatches

def count_column_mismatches(first, second):
    number_of_mismatches = 0

    for i in range(len(first)):
        if first[i][1] != second[i][3]:
            number_of_mismatches += 1

    return number_of_mismatches

input_file_path = "Ass1Output.txt"

with open(input_file_path, "r") as input_file:
    
    current_text = input_file.readline()

    puzzle_pieces = []
    current_number = " "

    row_size = 8
    column_size = 8
    delimiter = ' '

    for i in range(row_size):
        row = []

        current_text = input_file.readline()
        current_line = current_text.strip().split(delimiter)

        for current_number in current_line:
            row.append(current_number)

        puzzle_pieces.append(row)

    print(np.array(puzzle_pieces).reshape(8,8))
    number_of_mismatches = 0

    for i in range(row_size - 1):
        number_of_mismatches += count_row_mismatches(puzzle_pieces[i], puzzle_pieces[i + 1])

    for i in range(column_size - 1):
        first_column = [puzzle_pieces[j][i] for j in range(row_size)]
        second_column = [puzzle_pieces[j][i + 1] for j in range(row_size)]

        number_of_mismatches += count_column_mismatches(first_column, second_column)

print("Number of mismatches:", number_of_mismatches)
