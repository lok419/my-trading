from abc import ABC, abstractmethod

class TradingSubSystemModel(ABC):
    '''
        Abstract class defines all nescessary function to form a TradingSubSystem object
    '''

    @abstractmethod
    def set_start_date(self):
        pass

    @abstractmethod
    def set_end_date(self):
        pass

    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def backtest(self):
        pass

    @abstractmethod
    def optimize(self):
        pass

    @abstractmethod
    def performance(self):
        pass

    @abstractmethod
    def get_position(self):
        pass

    @abstractmethod
    def get_return(self):
        pass