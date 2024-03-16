from strategy_v3.ExecuteSetup import ExecuteSetup
from strategy_v3.Strategy import GridArithmeticStrategy, SimpleMarketMakingStrategy, StrategyModel

CONFIG_PATH = "strategy_v3/execute.json"

class StrategyFactory(object):
    '''
        Factory class to generate the strategy class. Reason of creating new class is to avoid circular dependency
    '''        
    def __init__(self):
        pass

    def get(self, strategy_id: str) -> StrategyModel:           
        cls = eval(ExecuteSetup.read_all()[strategy_id]['class'])
        params = ExecuteSetup(strategy_id).read()        
        strategy = cls(**params)
        return strategy
        



    

    