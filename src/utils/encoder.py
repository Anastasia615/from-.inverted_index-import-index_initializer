import numpy as np
from abc import ABC, abstractmethod
from collections.abc import MutableMapping
from tqdm import tqdm

class AbstractEncoder(ABC):
    """Абстрактный класс для кодировщиков."""

    @staticmethod
    @abstractmethod
    def encode(arr: np.array) -> np.array:
        """Кодирует массив чисел."""
        ...

    @staticmethod
    @abstractmethod
    def decode(arr: np.array) -> np.array:
        """Декодирует массив чисел."""
        ...


class EliasGammaEncoder(AbstractEncoder):
    """Реализация кодирования Элиаса-Гамма."""

    @staticmethod
    def encode(a):
        """
        Кодирует массив чисел с помощью кодирования Элиаса-Гамма.
        
        Args:
            a: Массив чисел для кодирования.
            
        Returns:
            Кортеж (закодированные данные, размер).
        """
        a = a.view(f'u{a.itemsize}')
        l = np.log2(a).astype('u1')
        L = ((l<<1)+1).cumsum()
        out = np.zeros(L[-1],'u1')
        for i in range(l.max()+1):
            out[L-i-1] += (a>>i)&1
        return np.packbits(out), out.size

    @staticmethod
    def decode(b, n):
        """
        Декодирует массив чисел из кодирования Элиаса-Гамма.
        
        Args:
            b: Закодированные данные.
            n: Размер закодированных данных.
            
        Returns:
            Декодированный массив чисел.
        """
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
    """Реализация кодирования Элиаса-Дельта."""

    @staticmethod
    def encode(a: np.array) -> tuple[np.array, int]:
        """
        Кодирует массив чисел с помощью кодирования Элиаса-Дельта.
        
        Args:
            a: Массив чисел для кодирования.
            
        Returns:
            Кортеж (закодированные данные, размер, первое число).
        """
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
        """
        Декодирует массив чисел из кодирования Элиаса-Дельта.
        
        Args:
            b: Закодированные данные.
            n: Размер закодированных данных.
            first_number: Первое число в последовательности.
            
        Returns:
            Декодированный массив чисел.
        """
        deltas = EliasGammaEncoder.decode(b, n)
        cumsum = np.cumsum(deltas)
        cumsum = np.insert(cumsum, 0, 0)
        
        return cumsum + np.ones_like(cumsum) * first_number


class EncodedInvertedIndex(MutableMapping):
    """
    Хранит значения инвертированного индекса в сжатом виде.
    """
    possible_encoders = {
        'gamma': EliasGammaEncoder,
        'delta': EliasDeltaEncoder
    }

    def __init__(self, inverted_index: dict[str, np.array], encoding_method='gamma'):
        """
        Инициализация сжатого инвертированного индекса.
        
        Args:
            inverted_index: Исходный инвертированный индекс.
            encoding_method: Метод кодирования ('gamma' или 'delta').
        """
        self.__dict = inverted_index
        self.encoder: AbstractEncoder = self.possible_encoders[encoding_method]()

        for key in tqdm(self.__dict):
            self.__dict[key] = self.__encode_value(self.__dict[key])

    def load_encoded_dict(self, encoded_dict):
        """Загружает закодированный словарь."""
        self.__dict = encoded_dict

    def get_encoded_dict(self):
        """Возвращает закодированный словарь."""
        return self.__dict

    def __encode_value(self, array: np.array) -> tuple:
        """Кодирует значение массива."""
        args = self.encoder.encode(array)
        return args
    
    def __decode_value(self, args: tuple) -> np.array:
        """Декодирует значение массива."""
        decoded_arr = self.encoder.decode(*args)
        return decoded_arr

    def __getitem__(self, key):
        """Получает декодированное значение по ключу."""
        decoded_value = self.__decode_value(self.__dict[key])
        return decoded_value
    
    def __setitem__(self, key, value):
        """Устанавливает закодированное значение по ключу."""
        encoded_value = self.__encode_value(value)
        self.__dict[key] = encoded_value

    def __delitem__(self, key):
        """Удаляет значение по ключу."""
        del self.__dict[key]

    def __iter__(self):
        """Возвращает итератор по ключам."""
        return iter(self.__dict)
    
    def __len__(self):
        """Возвращает количество элементов."""
        return len(self.__dict)
    
    def __str__(self):
        """Возвращает строковое представление словаря."""
        return str(self.__dict)
    
    def __repr__(self):
        """Возвращает представление объекта."""
        return '{}, D({})'.format(super(EncodedInvertedIndex, self).__repr__(), 
                                 self.__dict)

