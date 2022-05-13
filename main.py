import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import math



def cal_entropy(text):
    h = 0.0
    sum = 0
    letter = [0] * 26
    text = text.lower()
    for i in range(len(text)):
        if text[i].isalpha():
            letter[ord(text[i]) - ord('a')] += 1
            sum += 1
    if sum == 0:
        return 0
    else:
        for i in range(26):
            p = 1.0 * letter[i] / sum
            if p > 0:
                h += -(p * math.log(p, 2))
        return h


pattern_list = []


class Pattern:
    def __init__(self, _length, _numbers, _letter, _segmentation, _label):
        self.length=_length
        self.numbers=_numbers
        self.letter=_letter
        self.seg=_segmentation
        self.label=_label

    def return_data(self):
        return [self.length, self.numbers, self.seg, self.letter]

    def return_label(self):
        if self.label=="dga":
            return 1
        else:
            return 0


def change_data(domain):
    domains = domain.split(".")
    length = len(domains[0])
    numbers = 0
    for x in domains[0]:
        if x.isnumeric():
            numbers = numbers + 1
    letters = cal_entropy(domains[0])
    if "-" in domains[0]:
        seg = 1
    else:
        seg = 0
    return [length, numbers, letters, seg]


def ini_pattern(fliename):
    with open(fliename) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or line == "":
                break
            tokens = line.split(",")
            domain = tokens[0]
            label = tokens[1]
            domains = domain.split(".")
            length = len(domains[0])
            numbers = 0
            for x in domains[0]:
                if x.isnumeric():
                    numbers = numbers + 1
            letters = cal_entropy(domains[0])
            if "-" in domains[0]:
                seg = 1
            else:
                seg = 0
            pattern_list.append(Pattern(length,numbers,letters,seg,label))

test_list = []


def ini_test(fliename):
    with open(fliename) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or line == "":
                break
            test_list.append(change_data(line))


def main():
    print("initial data")
    ini_pattern("train.txt")
    featureMatrix = []
    labelList = []
    print("Initialize Matrix")
    for item in pattern_list:
        featureMatrix.append(item.return_data())
        labelList.append(item.return_label())
    print("Begin Training")
    clf = RandomForestClassifier(random_state=0)
    clf.fit(featureMatrix, labelList)
    print("pridict")
    ini_test("test.txt")
    output = clf.predict(test_list)
    i = 0
    print(output)
    with open("test.txt") as f:
        file = open("result.txt", 'a')
        for line in f:
            line = line.strip()
            if output[i] == 0:
                file.write(line + ",notdga\n")
            if output[i] == 1:
                file.write(line + ",dga\n")
            i = i + 1
        file.close()



if __name__ == '__main__':
         main()

