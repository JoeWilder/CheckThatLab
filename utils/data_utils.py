import os
import requests


def download(url: str, path: str = None):
    """Download file using the given url"""
    get_response = requests.get(url, stream=True)
    filename = url.split("/")[-1].split("?")[0]

    if path is not None:
        filename = os.path.join(path, filename)

    if os.path.exists(filename):
        return

    with open(filename, "wb") as f:
        for chunk in get_response.iter_content(chunk_size=1024):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
