
import numpy as np
import pandas as pd
from functools import reduce

from sklearn.base import TransformerMixin

import datetime
now = datetime.datetime.now()


class PipeYearsLog(TransformerMixin):
#    def __init__(self,)
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        Xlog = np.log1p(now.year-X)#
        return Xlog
    
    
class PipeMode(TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        for col in list(X):
            X[col].fillna("None", inplace=True)
        return X
    
class PipePasteFeature(TransformerMixin):

    def __init__(self, fetlist,fetToDelete):
        self.fetToDelete = fetToDelete
        self.fetlist=fetlist
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for fet in self.fetlist:     
            newf=fet[0]+fet[1]
            X[newf]=X[fet[0]].astype(str)+X[fet[1]].astype(str)
#            self.fetToDelete.add(fet[0])
#            self.fetToDelete.add(fet[1])

        X.drop(list(self.fetToDelete), axis=1,inplace=True)

class PipeCatToNumeric(TransformerMixin):
    def __init__(self,grp,target):
        self.grp=grp
        self.target=target
    def fit(self, X ,y=None):
        
      
        return self

    def transform(self, X):
       for selfgrp in self.grp:
#           print (selfgrp)
           a=X.groupby(selfgrp)[[self.target]].agg(['median']).reset_index()
           a.columns = [selfgrp,'t_median']
           a=a.sort_values('t_median')
           a[selfgrp]=a[selfgrp].astype('category')
           catname='cat'+selfgrp
           a[catname]=a['t_median'].astype('category').cat.codes+1
           a=a.set_index(selfgrp,drop=True)[catname]
           X=pd.merge(X, a.to_frame(), left_on=selfgrp,  right_index=True )
       X.sort_index(inplace=True)
       return X




class PipeJoin(TransformerMixin):

    def __init__(self, transformer_list):
        self.transformer_list = transformer_list

    def fit(self, X, y=None):
        for (name, t) in self.transformer_list:
            t.fit(X, y)
        return self

    def transform(self, X):
        Xts = [t.transform(X) for _, t in self.transformer_list]
        Xunion = reduce(lambda X1, X2: pd.merge(X1, X2, left_index=True, right_index=True), Xts)
        return Xunion





class PipeGetColumn(TransformerMixin):

    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X[self.cols]
        return X
    


class PipeFillZero(TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        Xz = X.fillna(value=0)
        return Xz

class PipeFillNone(TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        Xz = X.fillna(value="None")
        return Xz






