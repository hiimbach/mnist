# Number classification with MNIST

This is a side project in Swinburne Vietnam Lab. 

## Triton 
To run Triton Server:
`docker run --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 -v ${PWD}/triton:/models nvcr.io/nvidia/tritonserver:23.08-py3 tritonserver --model-repository=/models`

To run Triton Client:
```
docker run -it --rm --net=host -v ${PWD}:/workspace/ nvcr.io/nvidia/tritonserver:23.08-py3-sdk bash
pip install -r requirements
```