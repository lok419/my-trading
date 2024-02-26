from abc import abstractmethod, ABC

class AccountModel(ABC):
    
    @abstractmethod
    def get_position(self):
        pass

    @abstractmethod
    def place_order(self):
        pass    