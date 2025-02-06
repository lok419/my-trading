from strategy_v4.Model.Model import Model
from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor, StackingRegressor, VotingRegressor

class AdaBoostreg(Model):
    def __init__(self, **params):
        self.params = params    
        self.model = AdaBoostRegressor(**self.params)

    def fit(self, X, y):        
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
    
class BaggingReg(Model):
    def __init__(self, **params):
        self.params = params    
        self.model = BaggingRegressor(**self.params)

    def fit(self, X, y):        
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
    
class ExtraTreesReg(Model):
    def __init__(self, **params):
        self.params = params    
        self.model = ExtraTreesRegressor(**self.params)

    def fit(self, X, y):        
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)    
    
class GradientBoostingReg(Model):
    def __init__(self, **params):
        self.params = params    
        self.model = GradientBoostingRegressor(**self.params)

    def fit(self, X, y):        
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)   
    
class HistGradientBoostingReg(Model):
    def __init__(self, **params):
        self.params = params    
        self.model = HistGradientBoostingRegressor(**self.params)

    def fit(self, X, y):        
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

class RandomForestReg(Model):
    def __init__(self, **params):
        self.params = params    
        self.model = RandomForestRegressor(**self.params)

    def fit(self, X, y):        
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)