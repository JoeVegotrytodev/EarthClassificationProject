import torch
import numpy
from tqdm import tqdm
from torch import nn

from src.datasets.Data_set import Data_set as dataset
from src.device.Using_device import get_device
from src.dataloaders.TestSampler import TestSampler
from src.net.EarthClassNet import EarthClastNet
from src.log.WandBLog import WandBLog
from src.PictToClass import pic_to_class

loss_func = torch.nn.CrossEntropyLoss()
NUM_OF_EPOCH = 3
DEVICE = get_device()


def getOptimizer(model, learning_rate=0.01):
    """устанавливаем оптимайзер"""
    return torch.optim.SGD(model.parameters(), lr=learning_rate)


def train(model, optimizer, train_dataloader, device):
    """Обучение модели"""

    # берем пример и ответ
    for sample_train, answer_train in tqdm(train_dataloader):
        # переносим батчи на девайс
        sample_train, answer_train = sample_train.to(device), answer_train.to(device)
        # print("Размерность батча трейн пример ", sample_train.shape)
        # print("Размерность батча трейн ответ  ", answer_train.shape)

        # делаем предсказание и переносим его на девайс
        prediction = model(sample_train).to(DEVICE)
        # prediction = prediction.to(get_device())

        # print("\nответ сетки\n[bs, числа которые станут вероятностями]")
        # print(prediction.shape)
        # print(prediction)
        # print("\nответ из выборки")
        # print(y_train.shape)
        # print("\nSoftMax по ответу: ")
        # print(nn.Softmax()(y_pred))

        # получаем классы по изображениям
        answer_train = pic_to_class(answer_train)

        loss = loss_func(prediction, answer_train)
        # loss =  torch.nn.NLLLoss()(torch.log(y_pred), y_train) - это было NLLL(Log(Softmax))

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def validate(model, test_dataloader, wandb, val_loss, val_accuracy, device):

    for sample_test, answer_test in tqdm(test_dataloader):
        sample_test, answer_test = sample_test.to(device), answer_test.to(device)
        # print("Размерность батча тест пример ", sample_test.shape)
        # print("Размерность батча тест ответ  ", answer_test.shape)

        # print("\nответ из выборки")
        classes = pic_to_class(answer_test)

        # y_pred2 = earth_model(x_val.permute(0, 3, 1, 2))
        test_prediction = model(sample_test)
        # print("test_prediction shape = ", test_prediction.shape)
        # print("test_prediction Ответ на выборку = ", test_prediction)
        print("\nSoftMax по ответу: ")
        print(nn.Softmax()(test_prediction))

        loss = loss_func(test_prediction, classes)
        loss = loss.to(device)
        val_loss.append(loss.numpy())

        # val_accuracy.extend((torch.argmax(test_prediction, dim=-1) == pic_to_class(answer_test)).numpy().tolist())

        print("mean_val_loss = ", numpy.mean(val_loss))
        wandb.log({"mean val loss:": numpy.mean(val_loss)})


if __name__ == '__main__':
    print("device =", DEVICE, '\n')

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
    earth_model.to(DEVICE)

    # задаем оптимайзер
    optimizer = getOptimizer(earth_model)

    # инициализация лога
    wandb = WandBLog(earth_model)

    for epoch in range(NUM_OF_EPOCH):
        train(earth_model, optimizer, train_dataloader, DEVICE)

        val_loss = []
        val_accuracy = []
        with torch.no_grad():
            validate(earth_model, test_dataloader, wandb.getWandB(), val_loss, val_accuracy, DEVICE)
