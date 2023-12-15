import numpy
import torch

from src.device.Using_device import get_device


def pic_to_class(answerTensor):
    """метод для получения класса на основе картинки
    Имеем 3 класса по кол-ву белых пикселей:
    1 - мало пикселей белых
    2 - среднее кол-во бел пикселей
    3 - много белых пикселей"""

    # print("answerTensor тензор перед приведением его к 10-3 = ", answerTensor)
    # итерация по тензору. получается проходим по строкам тензора
    for i, x in enumerate(answerTensor):
        print("- - -")
        # print("Получили из тензора 1 \ 1024 \ 1024 = ", answerTensor.shape)

        # тензор приводим к списку, чтобы проще работать с ним
        tensor_as_list = answerTensor.tolist()
        # print("Привели тензор к списку")

        # Создадим счетчик пикселей и результирующи список
        colorCount = 0
        result = []

        # print("Размерность списка = ", numpy.asarray(tensor_as_list).shape)
        # print("Сам список = ", numpy.asarray(tensor_as_list))

        # print("проходим по списку 1, 1024, 1024 ?????")
        for onePicOfBatch in tensor_as_list:
            colorCount = 0
            # print("Проходим по 1 размерности")
            # print("onePicOfBatch len = ", len(onePicOfBatch))
            for rows in onePicOfBatch:
                # print("Проходим по 2 размерности")
                # print("rows len = ", rows)
                for cols in rows:
                    # print("Проходим по 3 размерности")
                    # print("cols len = ", cols)
                    for pixel in cols:
                        # к - значение пикселя
                        if pixel == 255:
                            colorCount = colorCount + 1
                            # print("colorCount = ", colorCount)
                    # print("кол-во пикселей после итерации 3 размерности = ", colorCount)

            if colorCount > 100000:
                # print("A lot of buildings more than 100_000 pix")
                result.append([0, 0, 1])
            elif colorCount > 10000:
                # print("Average num of  buildings 10_000 - 100_000 pix")
                result.append([0, 1, 0])
            else:
                # print("A few buildings  0 - 10_000 pix")
                result.append([1, 0, 0])

            # print("размерност листа с классом ответа = ", len(result))
            if len(result) == 10:
                break

        #     print("colorCount after rows = ", colorCount)
        # print("colorCount after onePicOfBatch = ", colorCount)

        # tResult = torch.Tensor(result)
        tResult = torch.Tensor(result).to(get_device())
        if len(result) == 10:
            break
    # print("Тензор после приведения к классу: \n", tResult)

    return tResult
