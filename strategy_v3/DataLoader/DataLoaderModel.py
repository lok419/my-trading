from abc import ABC, abstractmethod

class DataLoaderModel(ABC):
    
    @abstractmethod
    def load_price_data():
        pass
