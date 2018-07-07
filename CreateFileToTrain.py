
import numpy as np
import pandas as pd
from imputClass import imputByRandomForest
import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

trainOrg = pd.read_csv('../input/train.csv')
testOrg = pd.read_csv('../input/test.csv')


#train.drop(train[(train["GrLivArea"]>4000)&(train["SalePrice"]<300000)].index,inplace=True)
pd.concat([test, train]).isnull().sum().sort_values(ascending=False).head(40)
train.set_index('Id',inplace=True,drop=True)
test.set_index('Id',inplace=True,drop=True)

y= train['SalePrice']

train['LotFrontage'],test['LotFrontage']=imputByRandomForest('SalePrice','LotFrontage',train.copy(),test.copy()).imput()

list(test)


NEMBERS=["MasVnrArea", "BsmtUnfSF", "TotalBsmtSF", "GarageCars", "BsmtFinSF2", "BsmtFinSF1", "GarageArea"]
CAT1 = ["PoolQC" , "MiscFeature", "Alley", "Fence", "FireplaceQu", "GarageQual", "GarageCond", "GarageFinish", "GarageType", "BsmtExposure", "BsmtCond", "BsmtQual", "BsmtFinType2", "BsmtFinType1", "MasVnrType"]
CAT2 = ["MSZoning",   "Utilities", "Functional", "Electrical", "KitchenQual", "SaleType","Exterior1st", "Exterior2nd"]
#a=list(set(NEMBERS + CAT1 +CAT2))
OUTTER = list(set(test)-set(NEMBERS + CAT1 +CAT2)  )


#pd.concat([test, train])[cols].isnull().sum().sort_values(ascending=False).head(40)

from PipeLibClass import PipeJoin,PipeGetColumn,PipeCatToNumeric,PipeFillZero,PipeFillNone,PipeMode,PipeYearsLog,PipePasteFeature
from sklearn.pipeline import Pipeline

pipeline1 = Pipeline([
    ('features', PipeJoin([
        ('number', Pipeline([
            ('get', PipeGetColumn(NEMBERS))
           , ('number', PipeFillZero())
            
        ])),
     ('CAT1', Pipeline([
              ('get', PipeGetColumn(CAT1))
           , ('none', PipeFillNone())
        ]))
,
     ('CAT2', Pipeline([
              ('get', PipeGetColumn(CAT2))
           , ('STR1', PipeMode())

        ]))
,  ('OUTTER', Pipeline([
              ('get', PipeGetColumn(OUTTER))


        ]))
    ]))
])
pipeline1.fit(train)
train = pipeline1.transform(train)
test = pipeline1.transform(test)
target=['SalePrice']
#CAT3=["MSSubClass","MSZoning","Neighborhood","Condition1","BldgType","HouseStyle","Exterior1st","MasVnrType","ExterQual","BsmtExposure","Heating","SaleCondition","HeatingQC","KitchenQual","Functional","FireplaceQu","GarageType","PavedDrive","SaleType"]
CAT3=list(train.select_dtypes(include=['object']).columns)
CAT3+=['MSSubClass']
YEARS=["YearBuilt","YearRemodAdd","GarageYrBlt","YrSold"]

#trainOrg.unique()
df_dict = dict(zip([i for i in trainOrg.columns] , [pd.DataFrame(trainOrg[i].value_counts(), columns=[i]) for i in trainOrg.columns]))


FEATUR=['BedroomAbvGr','MSSubClass','BldgType','BsmtCond','BsmtFinType2','OverallQual','Foundation','KitchenQual','OverallCond']
COMBINE_FEATUR=list()
for i in range(0,8):
    for j in range(i,8):
       if(i!=j):
           COMBINE_FEATUR.append(([FEATUR[i],FEATUR[j]]))



#a.dtypes
#b=a.describe()
# ['BsmtHalfBath', 'BsmtFullBath']

train['type_df']=1
test['type_df']=2



OUTTER = list(set(test)-set(CAT3+YEARS )  )
a=train[CAT3]
train['SalePrice']=y
test['SalePrice']=np.nan
dfall=pd.concat([test,train])
dfall.sort_index(inplace=True)



dfall2=dfall[FEATUR]
obj=PipePasteFeature(COMBINE_FEATUR,FEATUR)
obj.fit(dfall2).transform(dfall2)

dfall=pd.concat([dfall,dfall2],axis=1)
#a.drop(list(obj.fetToDelete), axis=1,inplace=True)

CAT3+=list(dfall2)
    
pipeline2 = Pipeline([
    ('features', PipeJoin([

     ('CAT3', Pipeline([
              ('get', PipeGetColumn(list(set(CAT3 + target))))
           , ('vat', PipeCatToNumeric(CAT3,target[0]))
        ]))

,  ('YEAR', Pipeline([
              ('get', PipeGetColumn(YEARS))
             , ('YEAR', PipeYearsLog())     

        ]))
,  ('OUTTER', Pipeline([
              ('get', PipeGetColumn(OUTTER))


        ]))
    ]))
])



pipeline2.fit(dfall)
dfall = pipeline2.transform(dfall.copy())

#a=dfall[['catMSSubClass','MSSubClass']].sort_values(by="catMSSubClass")
a=dfall[['SalePrice']]
dfall.drop(list(set(CAT3 + target)), axis=1,inplace=True)
objects=list(dfall.select_dtypes(include=['object']).columns)

for col in objects:
    dfall[col]=dfall[col].astype(int)
train=dfall[dfall['type_df']==1]
test=dfall[dfall['type_df']==2]

list(dfall.select_dtypes(include=['object']))



test.to_pickle('test_1.pkl')
train.to_pickle('train_1.pkl')
y.to_pickle('y_1.pkl')
