#!/usr/bin/env python

"""
Author: Vojtěch Čoupek
Description: optimization range processing
Project: Weight-Sharing of CNN - Diploma thesis FIT BUT 2023
"""

def process_range_2n(search_rang:range, ensure:bool = False):
    """Takes a range and outputs list where only 2^N members are present.

    Args:
        search_rang (range): is the input range
        ensure (bool, optional): If the next 2^N value should be present.
            Example: range(1, 10) - if True, 16 will be the last item, if False
            8 will be the last item of result. Defaults to False.

    Returns:
        list: The output list with 2^N members.
    """

    result = []
    x = 2

    while x in search_rang:
        result.append(x)
        x *= 2 

    if ensure:
        result.append(x)

    return result


if __name__ == '__main__':
    print(process_range_2n(range(1, 121), True))