import os.path

import torch
from torchvision.io import read_image


class Data_set(torch.utils.data.Dataset):
    """Класс для создания Датасета обучающей выборки,
    наследуем Датасет торча и определяем в нем конструктор,
    __len__ - длину датасета,
    __getitem__ - получение элемента датасета по индексу"""

    def __init__(self, sample_path, answer_path, file_format, ds_size, transform=None):
        """@sample_path (string) - путь к файлу с объектами
        @answer_path (string) - путь к файлу с ответами
        @transform (callable, optional) - преобразованием, по умолчанию отуствует"""

        # self.sample_catalog = "S:/AI/Kursov/LEVIR-CD+/train/label/train_"
        # self.answer_catalog = "S:/AI/Kursov/LEVIR-CD+/train/label/train_"
        self.sample_catalog = sample_path
        self.answer_catalog = answer_path
        self.file_format = file_format
        self.ds_size = ds_size

    def __len__(self):
        ''' возвращает кол-во объектов в датасете'''
        # меняем число в зависимости от того нам модель обучать или посомтреть запускает ли вообще
        #         return 636
        return self.ds_size

    def __getitem__(self, index):
        ''' загружаем и возвращаем пример из датасета по переданному индексу
        index - опеределяет расположение на диске
        Основываясь на индексе, он идентифицируем местоположение изображения на диске,
        преобразует его в тензор с помощью read_image'''

        # Основываясь на индексе, он идентифицируем местоположение изображения на диске,
        sample_path = os.path.join(self.sample_catalog + str(index) + self.file_format)
        answer_path = os.path.join(self.answer_catalog + str(index) + self.file_format)

        # преобразуем изобраежение в тензор с помощью read_image,
        sample = read_image(sample_path).to(dtype=torch.float32)
        answer = read_image(answer_path).to(dtype=torch.float32)

        # Если необходимы преобразования
        # transform = transforms.Compose([
        #   transforms.PILToTensor() ])
        return sample, answer

    def printState(self):
        print("sample_catalog =", self.sample_catalog,
              "answer_catalog =", self.answer_catalog,
              "file_format =", self.file_format,
              "ds_size =", self.ds_size)
