"""
Developed by Benyamin Eslami

Model : the abstract interface. 
        Every real model (src/models/...) inherits from it and implements train / evaluate / test / predict.
ML    : what the front end uses. It is given a Model and only ever calls the interface, so it never depends on how a model works inside.
"""

from abc import ABC, abstractmethod

class Model(ABC):

    @abstractmethod
    def feature_engineering(self, train_data, test_data):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass

    @abstractmethod
    def test(self):
        pass


class ML:
    def __init__(self, model: Model):
        if not isinstance(model, Model):
            raise TypeError("model must inherit from Model (src/ML_interface.py)")
        self.model = model
    
    def feature_engineering(self, train_data, test_data):
        return self.model.feature_engineering(train_data, test_data)
    
    def train(self):
        return self.model.train()

    def evaluate(self):
        return self.model.evaluate()

    def test(self):
        return self.model.test()
