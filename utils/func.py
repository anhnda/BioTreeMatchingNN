
def get_insert_dict_index(d, k, offset=0):
    try:
        idx = d[k]
    except:
        idx = len(d)+offset
        d[k] = idx

    return idx

def get_index_dict(d,k, v = -1):
    try:
        v = d[k]
    except:
        pass
    return v