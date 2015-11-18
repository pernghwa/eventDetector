import string

table = string.maketrans("","")
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)

def remove_punc(s):
    if isinstance(s, unicode):
        return s.translate(remove_punctuation_map)
    elif isinstance(s, (str, string)):
        return s.translate(table, string.punctuation)
    else:
        raise("s is neither a string nor unicode")