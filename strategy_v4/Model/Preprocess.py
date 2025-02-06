from abc import ABC
import pandas as pd

class Preprocess(ABC):
    def __init__(self):
        pass

    def fit(self):
        pass

    def transform(self) -> pd.DataFrame:
        pass