from strategy_v4.Data.Data import DataLayer
from strategy_v4.config import DATA_LAYER

def run():
    dl_args = {key: value for key, value in vars(DATA_LAYER).items() if not key.startswith('_')}
    dl = DataLayer(**dl_args)
    dl.load()
    dl.process()
    dl.upload()

if __name__ == '__main__':
    run()