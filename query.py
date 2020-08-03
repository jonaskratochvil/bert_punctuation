import requests
import sys
import hyperparameters as hp

url = f"http://{hp.HOST}:{hp.PORT}/predict"

def read_text(filename):
    with open(filename, "r") as f:
        text = [t.strip() for t in f.readlines()]
    return text

text = read_text(sys.argv[1])
for line in text:
    r = requests.post(url, json={"text": f"{line}"}).json()

    if hp.LIVE_ASR:
        print(r)
    else:
        print(r["response"]["prediction"])
