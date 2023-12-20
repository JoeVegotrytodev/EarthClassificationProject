import torch
import numpy
from tqdm import tqdm
from torch import nn
import matplotlib.pyplot as plt

from src.datasets.Data_set import Data_set as dataset
from src.device.Using_device import get_device
from src.dataloaders.TestSampler import TestSampler
from src.net.EarthClassNet import EarthClastNet
from src.log.WandBLog import WandBLog
from src.PictToClass import pic_to_class
from metrics.accuracy import accuracy_calc

loss_func = torch.nn.CrossEntropyLoss()
NUM_OF_EPOCH = 2
DEVICE = get_device()
BATCH_SIZE = 10
# TRAIN_DATASET_SIZE = 636
TRAIN_DATASET_SIZE = 10
TEST_DATASET_SIZE = 10

TRAIN_SAMPLE_PATH_NB = "C:/Users/Admin/PycharmProjects/EarthClassificationProject/data/train_label/train_"
TRAIN_ANSWER_PATH_NB = "C:/Users/Admin/PycharmProjects/EarthClassificationProject/data/train_label/train_"
TEST_SAMPLE_PATH_NB = "C:/Users/Admin/PycharmProjects/EarthClassificationProject/data/test_label/train_"
TEST_ANSWER_PATH_NB = "C:/Users/Admin/PycharmProjects/EarthClassificationProject/data/test_label/train_"

TRAIN_SAMPLE_PATH_DT = "S:/AI/Kursov/LEVIR-CD+/train/label/train_"
TRAIN_ANSWER_PATH_DT = "S:/AI/Kursov/LEVIR-CD+/train/label/train_"
TEST_SAMPLE_PATH_DT = "S:/AI/Kursov/LEVIR-CD+/test/label/train_"
TEST_ANSWER_PATH_DT = "S:/AI/Kursov/LEVIR-CD+/test/label/train_"


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

        classes = pic_to_class(answer_test)
        # print("\nответ из выборки", classes)

        # y_pred2 = earth_model(x_val.permute(0, 3, 1, 2))
        test_prediction = model(sample_test)
        test_prediction = torch.nn.functional.normalize(test_prediction)
        print("\nSoftMax по ответу: ")
        res = nn.Softmax()(test_prediction)
        print(res)

        # новый код для вывода результата
        for_print = classes.tolist()
        res2 = res.tolist()
        temp = answer_test.to("cpu")
        counter = 0
        print("answer_test")
        for idx, img in enumerate(temp):
            img = torch.squeeze(img)
            plt.imshow(img)
            plt.suptitle(for_print[counter])
            plt.title(res2[counter])
            counter = counter + 1
            plt.show()

        loss = loss_func(test_prediction, classes)
        loss = loss.to("cpu")
        val_loss.append(loss.numpy())
        print("loss ", loss)
        wandb.log({"loss:": loss.numpy()})
        wandb.log({"mean val loss:": numpy.mean(val_loss)})

        accuracy = accuracy_calc(test_prediction, answer_test, BATCH_SIZE)
        wandb.log({"accuracy:": accuracy})
        wandb.log({"mean accuracy:": numpy.mean(accuracy)})


if __name__ == '__main__':
    print("device =", DEVICE, '\n')

    trainDataset = dataset(TRAIN_SAMPLE_PATH_DT,
                           TRAIN_ANSWER_PATH_DT,
                           ".png",
                           TRAIN_DATASET_SIZE)

    testDataset = dataset(TEST_SAMPLE_PATH_DT,
                          TEST_ANSWER_PATH_DT,
                          ".png",
                          TEST_DATASET_SIZE)

    # trainDataset.printState()
    # testDataset.printState()

    testSampler = TestSampler(testDataset, 638)
    # testSampler.printState()

    # Создаем даталоудеры
    train_dataloader = torch.utils.data.DataLoader(trainDataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(testDataset, batch_size=BATCH_SIZE, sampler=testSampler)

    # for batch in train_dataloader:
    #     print(batch[0].size())
    #     print(batch[0])
    #     break
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
