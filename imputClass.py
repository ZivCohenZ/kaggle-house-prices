# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 00:00:33 2018

@author: ziv
"""

#input numeric value by using random forest
import pandas as pd

import warnings
warnings.filterwarnings('ignore')


from sklearn.ensemble import RandomForestRegressor

class imputByRandomForest:
    def __init__(self,target_,FetureToImpute_,train_,test_):
        self.target=target_
        self.FetureToImpute=FetureToImpute_
        self.train=train_
        self.test=test_
        
    def imput(self):
    
        X=self.train.drop(([self.target]),axis=1)
        X=X.select_dtypes(exclude=["object"])
        X_train=X[X[self.FetureToImpute].isnull()==False]
        X_test=X[X[self.FetureToImpute].isnull()]
        
        y=X_train[self.FetureToImpute].values
        X_train.drop(([self.FetureToImpute]),axis=1, inplace=True)
        X_test.drop(([self.FetureToImpute]),axis=1, inplace=True)
        regr = RandomForestRegressor(random_state=0, n_jobs=-1)
        X_train=X_train.fillna(X_train.mean())
        X_test=X_test.fillna(X_test.mean())
        
        regr.fit(X_train, y)
        
        ypre=regr.predict(X_test)
        
         
        df=pd.concat([self.test,self.train.drop([self.target],axis=1)])
        df=df.select_dtypes(exclude=["object"])
        df=df[df[self.FetureToImpute].isnull()]
        df=df.drop(([self.FetureToImpute]),axis=1)
        
        df.sort_index(inplace=True)
        df=df.fillna(df.mean())
        ypre=regr.predict(df)
        
        
        df[self.FetureToImpute]=ypre
        self.test['type_df']=2
        self.train['type_df']=1
        
        a=pd.concat([self.test,self.train.drop([self.target],axis=1)])
        a.sort_index(inplace=True)
   
        b=a[a[self.FetureToImpute].isnull()]
        a=a[a[self.FetureToImpute].isnull()==False]
        
        
        
        b[self.FetureToImpute]=ypre
      
        c=pd.concat([a,b]).sort_index()#[self.FetureToImpute]
        ctrain=c[c['type_df']==1]
        ctest=c[c['type_df']==2]
        return ctrain[self.FetureToImpute],ctest[self.FetureToImpute]
