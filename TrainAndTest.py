"""

import pandas as pd
import numpy as np
from BayesianOptimization_XGB import rmsle_cv,xgb_model
from Stacking import BaseStacking

X = pd.read_pickle('../input/train_1.pkl')


y=pd.read_pickle('../input/y_1.pkl')
y=np.log1p(y)





model=xgb_model(train=X,y_train=y,random_state=11,num_rounds=3000,num_iter=25,eta=0.05,init_points=5)
model.best_RMSE
score2=model.TrainModel()


mxgb=model.GetModel()

score = rmsle_cv(mxgb,X,y,5,1)
print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))



model2=xgb_model(train=X,y_train=y,random_state=122,num_rounds=3000,num_iter=10,eta=0.05,init_points=5)
model2.best_RMSE
model.best_RMSE

#Xgboost score: 0.1160 (0.0089)


model3=xgb_model(train=X,y_train=y,random_state=122,num_rounds=3000,num_iter=10,eta=0.05,init_points=5)
model3.best_RMSE


averaged_models = BaseStacking(models = (model2.GetModel(), model.GetModel(),model3.GetModel()))
averaged_models
#averaged_models = BaseStacking(models = (model2.GetModel(), model.GetModel()))

score = rmsle_cv(averaged_models,X,y,5,15654)
print(" Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()
