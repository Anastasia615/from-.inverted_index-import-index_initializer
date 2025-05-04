import numpy as np
from abc import ABC, abstractmethod
from collections.abc import MutableMapping
from tqdm import tqdm

class AbstractEncoder(ABC):

    @staticmethod
    @abstractmethod
    def encode(arr: np.array) -> np.array:
        ...

    @staticmethod
    @abstractmethod
    def decode(arr: np.array) -> np.array:
        ...



class EliasGammaEncoder(AbstractEncoder):
    @staticmethod
    def encode(a):
        a = a.view(f'u{a.itemsize}')
        l = np.log2(a).astype('u1')
        L = ((l<<1)+1).cumsum()
        out = np.zeros(L[-1],'u1')
        for i in range(l.max()+1):
            out[L-i-1] += (a>>i)&1
        return np.packbits(out), out.size

    @staticmethod
    def decode(b,n):
        if len(b) == 0:
            return np.array([])
        b = np.unpackbits(b,count=n).view(bool)
        s = b.nonzero()[0]
        s = (s<<1).repeat(np.diff(s,prepend=-1))
        s -= np.arange(-1,len(s)-1)
        s = s.tolist() # list has faster __getitem__
        ns = len(s)
        def gen():
            idx = 0
            yield idx
            while idx < ns:
                idx = s[idx]
                yield idx
        offs = np.fromiter(gen(),int)
        sz = np.diff(offs)>>1
        mx = sz.max()+1
        out = np.zeros(offs.size-1,int)
        for i in range(mx):
            out[b[offs[1:]-i-1] & (sz>=i)] += 1<<i
        return out


class EliasDeltaEncoder(AbstractEncoder):
    @staticmethod
    def encode(a: np.array) -> tuple[np.array, int]:
        if len(a) == 0:
            return (a, 0, 0)
        if len(a) == 1:
            encoded, n = EliasGammaEncoder.encode(a)
            return (encoded, n, 0)
        a.sort()
        deltas = np.diff(a)
        gamma_encoded, n = EliasGammaEncoder.encode(deltas)
        
        return gamma_encoded, n, a[0]

    @staticmethod
    def decode(b: np.array, n: int, first_number: int) -> np.array:
        deltas = EliasGammaEncoder.decode(b, n)
        cumsum = np.cumsum(deltas)
        cumsum = np.insert(cumsum, 0, 0)
        

        return cumsum + np.ones_like(cumsum) * first_number

    

class EncodedInvertedIndex(MutableMapping):
    """
    Stores values of 
    """
    possible_encoders = {
        'gamma':EliasGammaEncoder,
        'delta':EliasDeltaEncoder
    }
    def __init__(self, inverted_index: dict[str, np.array], encoding_method='gamma'):
        self.__dict = inverted_index
        self.encoder: AbstractEncoder = self.possible_encoders[encoding_method]()

        for key in tqdm(self.__dict):
            self.__dict[key] = self.__encode_value(self.__dict[key])


    def load_encoded_dict(self, encoded_dict):
        self.__dict = encoded_dict

    def get_encoded_dict(self):
        return self.__dict


    def __encode_value(self, array: np.array) -> tuple:
        args = self.encoder.encode(array)
        return args
    
    def __decode_value(self, args: tuple) -> np.array:
        decoded_arr = self.encoder.decode(*args)
        return decoded_arr

    def __getitem__(self, key):
        decoded_value = self.__decode_value(self.__dict[key])
        return decoded_value
    
    def __setitem__(self, key, value):
        encoded_value = self.__encode_value(value)
        self.__dict[key] = encoded_value

    def __delitem__(self, key):
        del self.__dict[key]

    def __iter__(self):
        return iter(self.__dict)
    
    def __len__(self):
        return len(self.__dict)
    
    def __str__(self):
        """returns simple str of a dict, values will be encoded"""
        return str(self.__dict)
    
    def __repr__(self):
        return '{}, D({})'.format(super(EncodedInvertedIndex, self).__repr__(), 
                                  self.__dict)

