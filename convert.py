import os
with open('label_dict/vn.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        print(line)
        a, b = line.strip().split('\t')
        new_a = a+' '+b+'\n'
        with open('label_dict/vn_space.txt', 'a') as n:
            n.write(new_a)