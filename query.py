import requests
import sys

def read_text(filename):
    with open(filename, 'r') as f:
        text = [t.strip() for t in f.readlines()]
    return text

text = read_text(sys.argv[1])
for line in text:
    url = "http://127.0.0.1:9902/predict"
    r = requests.post(url,json={'text':line}).json()

    print(r["response"]["prediction"])
