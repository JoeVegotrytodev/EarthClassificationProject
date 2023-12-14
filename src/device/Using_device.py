import torch
def get_device():
    """Метод возвращающий актуальное устройство ЦП или ГПУ
    Если доступно ГПУ берем его
    иначе ЦПУ"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return device