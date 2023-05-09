#!/usr/bin/env python

"""
Author: Vojtěch Čoupek
Description: Implementation float32 to float16 or float8 precision reducer
Project: Weight-Sharing of CNN - Diploma thesis FIT BUT 2023
"""


import sys
sys.path.append('../code')
from utils.float_prec_reducer.Float8Reducer import Float8Reducer
import numpy as np

class FloatPrecReducer:
    def __init__(self, float8_significand=3):
        """inits the reducer object, also inits the necessary convertors.

        Args:
            float8_significand (int, optional): is the significand length of the float8 convertor. Defaults to 3.
        """

        self.float8_reducer = Float8Reducer(significand_len = float8_significand)
        self.pres_bytes = {
            'f4': 4,
            'f2': 2,
            'f1': 1,
        }

    def reduce(self, f32value:float, rtype:str='f4') -> float:
        """reduces precision of a given value to given reduction type.

        reduction types:
            - 'f4' - float32 (the original value)
            - 'f2' - float16
            - 'f1' - float8

        Args:
            f32value (float): is the value of which the precision will be reduced.
            rtype (str, optional): is the reduction type. Defaults to 'f4'.

        Raises:
            Exception: is unknown reduction type is given, an exception is raised.

        Returns:
            float: given value with reduced precision.
        """

        if rtype == 'f4':
            return f32value
        elif rtype == 'f2':
            return self.float8_reducer.reduce(f32value)
        elif rtype == 'f1':
            pass
        raise Exception(f'reduce error - unknown type [{rtype}]')

    def reduce_list(self, f32list:list, rtype:str='f4') -> list:
        """reduces precision of every values in a given list to a given reduction type
        while the original list type remains.

         reduction types:
            - 'f4' - float32 (the original value)
            - 'f2' - float16
            - 'f1' - float8

        Args:
            f32list (list): is the input list of which the precision will be reduced.
            rtype (str, optional): is the reduction type. Defaults to 'f4'.

        Raises:
            Exception: if unknown reduction type is given, an exception is raised.

        Returns:
            list: the list with reduced values.
        """

        if rtype == 'f4':
            return f32list
        elif rtype == 'f2':
            f16slist = [str(x) for x in np.float16(f32list.reshape(-1))]
            return np.array([float(x) for x in f16slist], dtype='f4').reshape(f32list.shape)
        elif rtype == 'f1':
            return self.float8_reducer.reduce_list(f32list)
        raise Exception(f'reduce_list error - unknown type [{rtype}]')

    def get_prec_bytes(self, rtype:str) -> int:
        """Tells the number of bytes needed to theoreticaly save a
        number with a given reduction type.

        Args:
            rtype (str): is the desired reduction type.

        Returns:
            int: number of bytes to save number of this reduction type.
        """

        return self.pres_bytes[rtype]


if __name__ == '__main__':
    tmp = FloatPrecReducer()

    array_to_reduce = np.array([1, 1.25, 2.5, 3.44])

    print(tmp.reduce_list(array_to_reduce))
    print(tmp.reduce_list(array_to_reduce, 'f2'))
    print(tmp.reduce_list(array_to_reduce, 'f1'))