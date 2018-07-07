House Prices: Advanced Regression Techniques

My solution for the House Prices competition hosted on Kaggle.

https://www.kaggle.com/c/house-prices-advanced-regression-techniques



files:
1. CreateFileToTrain.py:  
   preprocess and features engineer by using imputClass.py and  PipeLineClass.py

2. imputClass.py: 
   missing values with random forest
   
3. PipeLineClass.py: 
   class library to transform data frames   
   
4.TrainAndTest.py: 
   load files created in CreateFileToTrain.py and train + stack it by using BayesianOptimization_XGB.py, Stacking.py
   
5.BayesianOptimization_XGB.py:
  train and optimized xgboost using BayesianOptimization library
  
6. Stacking.py:
   Stacking class
