def find(ls, it):
    """
    Returns the first index of the first occurrance of item in ls.
    Returns -1 if no occurence of the item is found.
    ls:   list
    it: item
    """

    idx = -1
    for i in range(len(ls)):
        if it == ls[i]:
            idx = i
            break

    return idx

def list2csv(l, sep = ', '):

        csv = ''

        for item in l:
            if csv != '':
                csv = csv + ', '
            csv = csv + item
        return csv

if __name__ == "__main__":
    print(find(['a', 'b', 'b'],'b'))

    