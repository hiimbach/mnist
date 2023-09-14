# Number classification with MNIST

This is a side project in Swinburne Vietnam Lab. 

# Triton Server
To run Triton server, run:
```
docker run --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 -v ${PWD}/triton:/models nvcr.io/nvidia/tritonserver:23.08-py3 tritonserver --model-repository=/models 
```

# Triton Client
To run Django inside the Triton client, run:
```
docker run -it --rm -p 9966:9966 --net=host -v ${PWD}:/workspace/ nvcr.io/nvidia/tritonserver:23.08-py3-sdk 
```
Then inside the Triton client, run:
```
pip install -r requirements
python django/manage.py runserver 0.0.0.0:9966
```
The API then be hosted at: 
- Upload: [localhost:9966/mnist/upload](localhost:9967/mnist/upload)
- Get result: [localhost:9966/mnist/result](localhost:9967/mnist/result)