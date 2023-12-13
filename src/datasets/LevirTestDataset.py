import os.path
import torch
from torchvision.io import read_image


class levirTestDataset(torch.utils.data.Dataset):
    """@sample_dir (string) - директория со всеми картиниками
     @transform (callable, optional) - преобразованием которое применяем"""

    def __init__(self, transform=None):
        self.sample_catalog = "S:/AI/Kursov/LEVIR-CD+/test/A/train_"
        self.answer_catalog = "S:/AI/Kursov/LEVIR-CD+/test/label/train_"

    ''' возвращает кол-во объектов в датасете'''

    def __len__(self):
        #         return 347
        return 64

    ''' загружаем и возвращаем пример из датасета по переданному индексу
    # index - опеределяет расположение на диске
    # Основываясь на индексе, он идентифицируем местоположение изображения на диске,
    # преобразует его в тензор с помощью read_image'''

    def __getitem__(self, index):
        # # Основываясь на индексе, он идентифицируем местоположение изображения на диске,
        sample_path = os.path.join(self.sample_catalog + str(index) + ".png")
        answer_path = os.path.join(self.answer_catalog + str(index) + ".png")

        #         # преобразуем изобраежение в тензор с помощью read_image,
        sample = read_image(sample_path).to(dtype=torch.float32)
        answer = read_image(answer_path).to(dtype=torch.float32)

        #         transform = transforms.Compose([
        #             transforms.PILToTensor() ])
        return sample, answer