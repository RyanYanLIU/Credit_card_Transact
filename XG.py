import pandas as pd 
import numpy as np 
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import r2_score
from sklearn.datasets import load_diabetes 
import shap

dia = load_diabetes()
data, target = dia.data, dia.target 

col = [f'Feature_{i}' for i in range(1, 11)]
df = pd.DataFrame(data, columns = col)
df['target'] = target 

#Use xgboost model
X, y = df.iloc[:, :-1], df.iloc[:, -1]
md = XGBRegressor(random_state = 123)

#Train test split
params = {'objective': ['reg:linear', 'reg:squarederror'],
            'booster': ['gbtree', 'gblinear'],
            'n_estimators': [i for i in range(8, 33, 2)],
            'max_depth': [i for i in range(0, 50, 3)],
            'learning_rate': [.1, .3, .5, .7, .9],
            'subsample': [.1, .2, .4, .6, .7],
            'reg_lambda': [.1, .3, .6]}
rscv = RandomizedSearchCV(md, param_distributions = params, cv = 5, scoring = 'neg_mean_squared_error', 
                         n_jobs = -1, random_state = 10)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .25, random_state = 10)
rscv.fit(X_train, y_train)
#print(f'Best Model is: {rscv.best_estimator_}\nBest Error is: {rscv.best_score_}')
#Model Training
model = rscv.best_estimator_
model.fit(X_train, y_train, eval_set = [(X_test, y_test)], early_stopping_rounds = 50, eval_metric = ['logloss'])

##Casual Inference
shaps = shap.TreeExplainer(model, X_test)
shap_value = shaps.shap_values(X_test)
shap.initjs()
shap.summary_plot(shap_values = shap_value, features = X_test, feature_names = X.columns)