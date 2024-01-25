import csv
file_1 = open("./solution(1).csv", "r")
file_2 = open("./solution_final.csv")

reader_1 = csv.reader(file_1)
reader_2 = csv.reader(file_2)

mistake_counter = 0
for line_1, line_2 in zip(reader_1, reader_2):
    for  e_1, e_2 in zip(line_1, line_2):
        if e_1 != e_2:
            mistake_counter += 1
            print(e_1, e_2, line_1, line_2)

print(mistake_counter)
        