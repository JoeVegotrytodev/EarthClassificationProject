import torch
from torch.utils.data import SequentialSampler


class TestSampler(torch.utils.data.SequentialSampler):
    """Мой семплер, чтобы получать числа для тестовой выборки с номера 638 и далее"""

    def __init__(self, data_source, start_num):
        """конструктор
        @data_source - источник данных dataset to sample from
        @start_num - номер с которого начнем перебирать значения
        """
        self.data_source = data_source
        self.len = len(data_source)
        # print("len = ", self.len)
        self.start_num = start_num
        # print("start_num = ", self.start_num)

    def __iter__(self):
        """возвращает итератор выборок в этом наборе данных"""
        return iter(range(self.start_num, self.start_num + self.len))

    def __len__(self):
        return self.len

    def printState(self):
        print("len =", self.len,
              "start_num =", self.start_num)
