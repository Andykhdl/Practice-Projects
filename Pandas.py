#from collections import Counter
s = "Hello Anmol How are you!"
def count_letters(s):
    a = dict(sorted({char: s.count(char) for char in s}.items()))
    return (a)
print(count_letters(s))