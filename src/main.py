import torch

from src.datasets.Data_set import Data_set as dataset
from src.device.Using_device import get_device
from src.dataloaders.TestSampler import TestSampler
from src.net.EarthClassNet import EarthClastNet
from src.log.WandBLog import WandBLog

loss_func = torch.nn.CrossEntropyLoss()
def setOptimizer(model):
    return torch.optim.SGD(model.parameters(), lr = 0.01)

if __name__ == '__main__':
    current_device = get_device()
    print("device =", current_device, '\n')

    trainDataset = dataset("C:/Users/Admin/PycharmProjects/EarthClassificationProject/data/train_label/train_",
                           "C:/Users/Admin/PycharmProjects/EarthClassificationProject/data/train_label/train_",
                           ".png",
                           100)

    testDataset = dataset("C:/Users/Admin/PycharmProjects/EarthClassificationProject/data/test_label/train_",
                          "C:/Users/Admin/PycharmProjects/EarthClassificationProject/data/test_label/train_",
                          ".png",
                          100)

    # trainDataset.printState()
    # testDataset.printState()

    testSampler = TestSampler(testDataset, 638)
    # testSampler.printState()

    # Создаем даталоудеры
    train_dataloader = torch.utils.data.DataLoader(trainDataset, batch_size=10)
    test_dataloader = torch.utils.data.DataLoader(testDataset, batch_size=10, sampler=testSampler)

    # for batch in train_dataloader:
    #     print(batch[0].size())
    #     print(batch[0])
    #     break
    #
    # for batch in test_dataloader:
    #     print(batch[0].size())
    #     print(batch[0])
    #     break

    # создаем модель и переносим на девайс
    earth_model = EarthClastNet()
    earth_model.to(get_device())

    # задаем оптимайзер
    optimizer = setOptimizer(earth_model)

    #инициализация лога
    log = WandBLog(earth_model)