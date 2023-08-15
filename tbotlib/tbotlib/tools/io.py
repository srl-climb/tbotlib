import pickle
import os

def isave(obj, path: str = '', **kwargs) -> None:

    path = ipath(path, **kwargs)

    # save obj
    print('saving to ' + path + '...')
    pickle.dump(obj, open(path, 'wb'))


def iload(path: str, **kwargs):

    path = ipath(path, overwrite = True, **kwargs)

    #load obj
    print('loading from ' + path + '...')
    return pickle.load(open(path, 'rb'))


def ipath(path: str = '', default_dir: str = '', default_name: str = 'default', default_ext: str = 'p', overwrite: bool = False) -> str:

    # split path in directory, name, extension
    dir, base = os.path.split(path)
    name, ext = os.path.splitext(base)

    # defaults
    if dir == '':
        dir = default_dir
    if name == '':
        name = default_name
    if ext == '':
        ext = '.' + default_ext

    # if the path exsists and overwriting is not desired, add number
    path  = os.path.join(dir, name + ext)
    n = 1
    while os.path.isfile(path) and not overwrite:
        path = os.path.join(dir, name + '_' + str(n) + ext)
        n += 1

    return os.path.normpath(path)

