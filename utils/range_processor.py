def process_range_2n(search_rang:range, ensure:bool = False):
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