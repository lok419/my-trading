from abc import abstractmethod, ABC

class ExecutorModel(ABC):

    @abstractmethod
    def set_portfolio(self):
        pass

    @abstractmethod
    def execute(self):    
        pass    

    @abstractmethod
    def get_position(self):
        pass

    @abstractmethod
    def get_order_history(self):
        pass