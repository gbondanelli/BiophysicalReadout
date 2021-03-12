import pickle

def open_pickle(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj

def save_pickle(filename, objlist):
    with open(filename, 'wb') as f:
        pickle.dump(objlist, f)

##

