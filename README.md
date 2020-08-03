## Punctuation model for live ASR
To build a model from Dockerfile run, note that you have to fill in valid S3 credentials to Dockerfile to download a pre-trained model. Then run:
```
docker build --no-cache -f Dockerfile -t punctuation_model:api .
```

To start the flask server run:
```
sudo docker run -p 9901:9901 -ti punctuation_model:api python3 app_flask.py --host "127.0.0.1" --port "9901"
```
