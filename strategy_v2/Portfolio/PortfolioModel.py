from abc import abstractmethod, ABC

class PortfolioModel(ABC):
    '''
        Abstract class defines all nescessary function to form a Portfolio object
    '''        
           
    @abstractmethod
    def set_start_date(self):
        pass
                
    @abstractmethod
    def set_end_date(self):
        pass
    
    @abstractmethod
    def backtest_subsystems(self):
        pass

    @abstractmethod
    def backtest(self):
        pass
    
    @abstractmethod
    def optimize(self):
        pass

    @abstractmethod
    def get_weights(self):
        pass       
    
    @abstractmethod
    def get_position(self):
        pass

    @abstractmethod
    def get_position_shs(self):
        pass
    
    @abstractmethod
    def get_return(self):
        pass

    @abstractmethod
    def rebalance(self):
        pass

    @abstractmethod
    def get_position_for_trade(self):
        pass