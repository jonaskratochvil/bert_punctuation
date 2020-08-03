import requests
import sys
import hyperparameters as hp

def read_text(filename):
    with open(filename, "r") as f:
        text = [t.strip() for t in f.readlines()]
    return text

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="checkpoints2/JasperDecoderForCTC-STEP-516250.pt")
    parser.add_argument("--port", default="Port number")
    return parser.parse_args() 

if __name__ == "__main__":
    args = get_args()
    url = f"http://{args.host}:{args.port}/predict"
    text = read_text(sys.argv[1])
    for line in text:
        r = requests.post(url, json={"text": f"{line}"}).json()
        if hp.LIVE_ASR:
            print(r)
        else:
            print(r["response"]["prediction"])
