import itertools

class Ring:
    """
    Ring datatype.
    https://codereview.stackexchange.com/a/85317
    """

    def __init__(self, items) -> None:
        
        self._items = list(items)

    def __iter__(self) -> itertools.cycle:
        
        return itertools.cycle(self._items)

    def _from_integer_index(self, idx) -> int:
        
        if not isinstance(idx, int):
            
            raise TypeError("Ring indices must be integers, not {}".format(type(idx)))

        if not len(self._items):
            
            raise IndexError("Indexing empty ring")

        return idx % len(self._items) #module

    def __getitem__(self, idx):
        
        if isinstance(idx, slice):
            
            step = 1 if idx.step is None else idx.step
            
            return [self[i] for i in range(idx.start, idx.stop, step)]

        return self._items[self._from_integer_index(idx)]

    def __setitem__(self, idx, value):
        
        self._items[self._from_integer_index(idx)] = value

    def __delitem__(self, idx):
        
        del self._items[self._from_integer_index(idx)]

    def __repr__(self):
        
        return "Ring({})".format(self._items)

    def index(self, value):

        return self._items.index(value)


if __name__ == '__main__':

    r = Ring([1,2,3,4])
    print(r)
    print(r[3])
    print(r[4])