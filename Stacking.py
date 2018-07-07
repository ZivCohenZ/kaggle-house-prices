"""
Created on Fri Jun 29 12:37:53 2018

@author: ziv
"""
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
#import pandas as pd
import numpy as np
class BaseStacking(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models        

    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
           
        for model in self.models_:
            model.fit(X, y)

        return self
    

    def predict(self, X):
        predictions = np.column_stack([ model.predict(X) for model in self.models_])
        return np.mean(predictions, axis=1)   

