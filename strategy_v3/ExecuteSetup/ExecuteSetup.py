import json

CONFIG_PATH = "strategy_v3/ExecuteSetup/execute_setup.json"

class ExecuteSetup(object):
    '''
        A Class which wraps the strategy hyperparameters and provide function to read and update the parameters
    '''        
    def __init__(self, strategy_id:str):
        self.strategy_id = strategy_id
        
    @staticmethod
    def read_all() -> dict:
        with open(CONFIG_PATH, "r") as file:
            config = json.load(file)
        return config

    def read(self) -> dict:
        config = ExecuteSetup.read_all()
        return config[self.strategy_id]
    
    def update(self, params: dict):
        config = ExecuteSetup.read_all()
        config[self.strategy_id] = params
        with open(CONFIG_PATH, "w") as file:
            json.dump(config, file, indent=4)