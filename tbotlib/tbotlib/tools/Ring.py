import itertools

class Ring:
    """
    Ring datatype.
    https://codereview.stackexchange.com/a/85317
    """

    def __init__(self, values) -> None:
        
        self._values = list(values)

    def __iter__(self) -> itertools.cycle:
        
        return itertools.cycle(self._values)

    def _from_integer_index(self, idx) -> int:
        
        if not isinstance(idx, int):
            
            raise TypeError("Ring indices must be integers, not {}".format(type(idx)))

        if not len(self._values):
            
            raise IndexError("Indexing empty ring")

        return idx % len(self._values) #module

    def __getitem__(self, idx):
        
        if isinstance(idx, slice):
            
            step = 1 if idx.step is None else idx.step
            
            return [self[i] for i in range(idx.start, idx.stop, step)]

        return self._values[self._from_integer_index(idx)]

    def __setitem__(self, idx, value):
        
        self._values[self._from_integer_index(idx)] = value

    def __delitem__(self, idx):
        
        del self._values[self._from_integer_index(idx)]

    def __repr__(self):
        
        return "Ring({})".format(self._values)
    
    def values(self) -> list:

        return self._values
        
    def index(self, value):

        return self._values.index(value)


if __name__ == '__main__':

    r = Ring([1,2,7,4])
    print(r.values())
    print(r[3])
    print(r[4])
    print(r.index(2))
    print(r._values)