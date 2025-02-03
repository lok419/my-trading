from strategy_v4.Evaluate.Evaluate import Evaluate

def run():
    eval = Evaluate()
    eval.load()
    eval.eval()
    eval.upload()

if __name__ == '__main__':
    run()