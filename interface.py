from main.train import train
from main.valid import valid
import os



if __name__ == "__main__":
    device_id = "cuda:0"
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    train(device_id=device_id)
    # valid(device_id=device_id)
    