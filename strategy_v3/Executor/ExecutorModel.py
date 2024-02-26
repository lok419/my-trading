from abc import abstractmethod, ABC

class ExecutorModel(ABC):

    @abstractmethod
    def set_logger(self):
        pass

    @abstractmethod
    def place_order(self):
        pass

    @abstractmethod
    def cancel_order(self):
        pass
    
    @abstractmethod
    def get_all_orders(self):
        pass

    @abstractmethod
    def fill_orders(self):
        pass

    @abstractmethod
    def add_trading_fee(self):
        pass
