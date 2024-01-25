import csv


def display(a):
    for e in a:
        print(e)

def checker(a1, a2):
    correct = 0
    incorrect = 0
    count = 1
    for e in zip(a1, a2):
        for o in zip(e[0], e[1]):
            count += 1
            if o[0] == o[1]:
                correct += 1
            else:
                print(o)
                print(e)
                incorrect += 1

    return (correct, incorrect, count, correct/count)
with open("labels_1.csv", "r") as file_1:
    with open("./Dataset/Train/Grid_labels.csv", "r") as file_2:
        reader_1 = csv.reader(file_1)
        predictions = []
        for line in reader_1:
            predictions.append(line)
        reader_2 = csv.reader(file_2)
        check = []
        for index, lines in enumerate(reader_2):
            if index == 0:
                continue
            check.append(lines[1:10])

print(checker(predictions, check))