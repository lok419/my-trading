from abc import ABC, abstractmethod

class StrategyModel(ABC):    

    @abstractmethod
    def set_strategy_id():
        pass

    @abstractmethod
    def set_data_loder():
        pass

    @abstractmethod
    def set_executor():
        pass

    @abstractmethod
    def load_data():
        pass

    @abstractmethod
    def execute():
        pass

    @abstractmethod
    def run():
        pass

    @abstractmethod
    def cancel_all_orders():
        pass

    @abstractmethod
    def is_delta_neutral():
        pass

    @abstractmethod
    def close_out_positions():
        pass
