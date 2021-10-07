# homework.py
# parse atom list and print periodic table
# Data: 2021. 09. 05.
# Author: Junho Lee

def atom_parse(string):
    atom_lst = []
    for s in string:
        if s.islower():
            atom_lst[-1] = atom_lst[-1]+s
        else:
            atom_lst.append(s)
    for i in range(len(atom_lst)):
        if len(atom_lst[i]) == 1:
            atom_lst[i] = ' '+atom_lst[i]

    return atom_lst


def print_periodic_table(atom_lst):
    period = [2, 10, 18, 36, 54]
    periodic_table = ''
    i = 0
    j = 0
    for a in atom_lst:
        if i < period[j]:
            if period[j] == 2 and i == 0:
                periodic_table = periodic_table + a + '   '*16
            elif period[j] in [10, 18] and i == period[j-1]+1:
                periodic_table = periodic_table + ' ' + a + '   '*10
            else:
                periodic_table = periodic_table + ' ' + a
        else:
            periodic_table = periodic_table + '\n' + a
            j = j+1
        i = i+1
    periodic_table = periodic_table + '\n'*(4-j)
    print(periodic_table)
    return


if __name__ == "__main__":
    print_periodic_table(atom_parse(input("input squeezed element names:\n")))
