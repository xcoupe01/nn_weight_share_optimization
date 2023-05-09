#!/usr/bin/env python

"""
Author: Vojtěch Čoupek
Description: Implementation float32 to float8 precision reducer
Project: Weight-Sharing of CNN - Diploma thesis FIT BUT 2023
"""

import numpy as np

BIT_LEN = 8

class Float8Reducer:
    def __init__(self, significand_len:int=3, special_numbers:bool=True, skip_special:bool=True) -> None:
        """ints the reducer object - creates the reduction table by a given set of parameters.

        Args:
            significand_len (int, optional): is the significand length. Defaults to 3.
            special_numbers (bool, optional): If true, the special values by minifloat are considered. Defaults to True.
            skip_special (bool, optional): iF true, the special values by minifloat are removed. Defaults to True.

        Raises:
            Exception: if the significand length is higher than 6 - no more room for exponent and sign.
        """

        if significand_len > BIT_LEN - 2:
            raise Exception('Float8Reducer init error - cannot represent that big significand')

        self.significand_len = significand_len
        self.convert_table = []
        self.bits = BIT_LEN
        
        # compute all the 8-bit numbers and get their float representations
        num = [False for _ in range(BIT_LEN)]
        for i in range(pow(2, BIT_LEN)):
            converted_num = bit2float(num, significand_len, special_numbers, skip_special)
            if converted_num is not None:
                self.convert_table.append(converted_num)
            num = increment_binary(num)
        
        self.convert_table.sort(key=sort_float)
        self.convert_table = np.array(self.convert_table)

    def reduce(self, f32value:float) -> float:
        """reduces the precision of given float32 value

        Args:
            f32value (float): is the value of which the precision should be reduced.

        Returns:
            float: is the value with reduced precision.
        """

        idx = (np.abs(self.convert_table - f32value)).argmin()
        return self.convert_table[idx]

    def reduce_list(self, f32list:list) -> list:
        """reduces the precision of all numbers in a given list.

        Args:
            f32list (list): is the input float32 list to reduce the presision in.

        Returns:
            list: the given list with reduced precision.
        """

        return np.array([self.reduce(x) for x in f32list.reshape(-1)], dtype='f4').reshape(f32list.shape)

def increment_binary(num:list) -> list:
    index = 0
    
    # set first bit if possible
    if not num[index]:
        num[0] = True
        return num

    # propagation of bit
    while index < len(num) and num[index]:
        num[index] = False
        index += 1

    if index < len(num):
        num[index] = True
    
    return num

def bit2float(num:list, significand_len:int, special_numbers:bool, skip_special:bool) -> float:
    """Converts a bool string with bit like representation into float with a given settings.

    The conversion is done by following algorithm:

    when exponent contains only '0': 
        append significand by 0
    else:
        append significand by 1
    (-1)^sign * significand * 2^exponent

    Args:
        num (list): is the input bool list encoding the bit like number.
        significand_len (int): is the significand length in the bit representation.
        special_numbers (bool): to consider special values defined by minifloat.
        skip_special (bool): to not return the special values defined by minifloat.

    Returns:
        float: the float representation of given bit representation.
    """

    exponent_bits = num[1:len(num) - significand_len]
    significand_bits = num[len(num) - significand_len:]

    # special numbers handling
    if special_numbers and all(exponent_bits):
        if skip_special:
            return
        if any(significand_bits):
            return float('nan')
        else:
            return float('inf') if num[0] else - float('inf')

    # computing the differend float parts
    significand_append = [True] if any(exponent_bits) else [False]
    exponent = bit2int(exponent_bits)
    significand = bit2significand(significand_append + significand_bits)

    return (-1 if num[0] else 1) * significand * pow(2, exponent)

def bit2int(num:list) -> int:
    """converts bool list representing a bit encoded number
    to integer value.

    Args:
        num (list): is the input bool list encoding the bit like number.

    Returns:
        int: the integer representation of a given number.
    """

    multiplier = 1
    acc = 0
    for i in range(len(num)):
        acc += multiplier if num[i] else 0
        multiplier *= 2
    return acc

def bit2significand(num:list) -> float:
    """converts bool list representing a bit encoded number
    to significand.

    Args:
        num (list): is the input bool list encodig the bit like number.

    Returns:
        float: significand (fraction)
    """

    multiplier = pow(2, -1)
    acc = 0
    for i in range(len(num)):
        acc += multiplier if num[i] else 0
        multiplier /= 2
    return acc

def sort_float(value:float) -> float:
    """Used to sort float numbers, with the special values

    Args:
        value (float): is the input value to sort

    Returns:
        float: the sort representation
    """

    if np.isnan(value):
        return - float('inf')
    return value


if __name__ == '__main__':
    tmp = Float8Reducer()
    print(tmp.convert_table)
    print(len(tmp.convert_table))
    print(tmp.reduce_list([69, 420, 6969]))