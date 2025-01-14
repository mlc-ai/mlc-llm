import argparse
import re
from pathlib import Path

import requests


def find_urls_in_file(file_path):
    with open(file_path, "r") as file:
        content = file.read()

    # Regular expression pattern to match URLs
    url_pattern = re.compile(
        r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    )

    # Find all matches of URLs in the content
    urls = re.findall(url_pattern, content)
    return [url.strip(">") for url in urls]


def main():
    parser = argparse.ArgumentParser(description="Check validity of links in documentation")
    parser.add_argument("--directory", type=str, default="docs", help="Directory of documentation.")
    args = parser.parse_args()

    # traversal the directory and find all rst files
    doc_directory = Path(args.directory)
    for file_path in doc_directory.glob("**/*.rst"):
        print("Checking {}...".format(file_path))
        for url in find_urls_in_file(file_path):
            try:
                r = requests.get(url)
                if r.status_code == 404:
                    print("404 not found: {}".format(url))
            except Exception as e:
                print("Error connecting {}, error: {}".format(url, e))


if __name__ == "__main__":
    main()
