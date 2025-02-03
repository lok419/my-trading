from abc import ABC
from pandas import Series

class Model(ABC):

    def __init__(self):
        pass

    def fit(self):
        pass

    def predict(self) -> Series:
        pass