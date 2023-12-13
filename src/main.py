import torch

from src.datasets.LevirTrainDataset import LevirTrainDataset


def get_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return device

# def toDevice(device):


if __name__ == '__main__':
    print(get_device())

    trainDataset = LevirTrainDataset()

