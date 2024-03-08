import json
import inspect

CONFIG_PATH = "strategy_v3/execute.json"

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
        params = config[self.strategy_id]
        del params['class']
        return params

    def update(self, key, value, cls: type):
        update_args = {arg.name: arg.annotation for arg in inspect.signature(cls.__init__).parameters.values()}
        assert key in update_args, f"key {key} is not property of strategy object"

        # cast the value to input type
        value = update_args[key](value)        
        config = ExecuteSetup.read_all()        
        config[self.strategy_id][key] = value
        with open(CONFIG_PATH, "w") as file:
            json.dump(config, file, indent=4)