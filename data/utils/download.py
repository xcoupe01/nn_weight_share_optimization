import requests
import os
import sys

CHUNK_SIZE = 8192

def nice_file_size(size:int):
    file_size_types = ['B', 'KB', 'MB', 'GB', 'TB']
    type_i = 0
    while size > 1000 and type_i < (len(file_size_types) - 1):
        size /= 1000
        type_i += 1

    return f'{size:0.01f}{file_size_types[type_i]}'


# source: https://stackoverflow.com/questions/15644964/python-progress-bar-and-downloads
def download_file(url:str, filepath:str=None) -> str:
    
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