#!/usr/bin/env python

"""
Author: Vojtěch Čoupek
Description: Dataset donwloading utility
Project: Weight-Sharing of CNN - Diploma thesis FIT BUT 2023
Inspired by: https://stackoverflow.com/questions/15644964/python-progress-bar-and-downloads 
"""

import requests
import os
import sys

CHUNK_SIZE = 8192

def nice_file_size(size:int) -> str:
    """Returns nice string representation of the file size.

    Args:
        size (int): is the size wanted to be converted into the nice size.
        Asumed that number of bytes is given and converted to B, MB ect sizes.

    Returns:
        str: the converted file size.
    """

    file_size_types = ['B', 'KB', 'MB', 'GB', 'TB']
    type_i = 0
    while size > 1000 and type_i < (len(file_size_types) - 1):
        size /= 1000
        type_i += 1

    return f'{size:0.01f}{file_size_types[type_i]}'


# source: https://stackoverflow.com/questions/15644964/python-progress-bar-and-downloads
def download_file(url:str, filepath:str=None) -> str:
    """Downloads large file from the internet.

    Args:
        url (str): Is the url of the file wanted to be downloaded.
        filepath (str, optional): Is the folder path for the file to be downloaded to. Defaults to None.

    Returns:
        str: The filepath of the downloaded file.
    """
    
    filename = os.path.join(filepath if filepath is not None else '', url.split('/')[-1])

    with open(filename, "wb") as f:
        print("Downloading %s" % filename)
        response = requests.get(url, stream=True)
        total_length = response.headers.get('content-length')

        if total_length is None: # no content length header
            f.write(response.content)
        else:
            dl = 0
            total_length = int(total_length)
            for data in response.iter_content(chunk_size=4096):
                dl += len(data)
                f.write(data)
                done = int(50 * dl / total_length) 
                sys.stdout.write(f"\r [{'=' * done}{' ' * (50-done)}] {nice_file_size(dl)}/{nice_file_size(total_length)} {(dl/total_length) * 100:0.1f}%{' ' * 6}")
                sys.stdout.flush()

    return filename