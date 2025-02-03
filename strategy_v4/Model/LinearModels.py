from abc import ABC
from strategy_v4.Model.Model import Model
from sklearn.linear_model import Lasso, Ridge, LinearRegression, ElasticNet

class LassoReg(Model):
    def __init__(self, **params):
        self.params = params    
        self.model = Lasso(**self.params)

    def fit(self, X, y):        
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

class RidgeReg(Model):
    def __init__(self, **params):
        self.params = params    
        self.model = Ridge(**self.params)

    def fit(self, X, y):        
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

class LinearReg(Model):
    def __init__(self, **params):
        self.params = params    
        self.model = LinearRegression(**self.params)

    def fit(self, X, y):        
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)    

class ElasticNetReg(Model):
    def __init__(self, **params):
        self.params = params    
        self.model = ElasticNet(**self.params)

    def fit(self, X, y):        
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)  