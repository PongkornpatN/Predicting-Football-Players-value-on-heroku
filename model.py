# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import math
from sklearn.metrics import mean_squared_error

dataset = pd.read_csv('batch.csv')

x_att = dataset[['age',
 'International Reputation',
 'Wage(kGBP)',
 'Best Overall Rating','Potential']]

y = dataset['fee_cleaned']
name_att= ['const',
           'age',
           'International Reputation',
           'Wage(kGBP)',
           'Best Overall Rating',
           'Potential']
X_train_att, X_test_att, y_train_att, y_test_att = train_test_split(
    x_att, y, test_size=0.2, random_state=13
)
#Splitting Training and Test Set
#Since we have a very small dataset, we will train our model with all availabe data.

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

GBR_model_att = GradientBoostingRegressor(n_estimators=20)
# define the evaluation procedure
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate the model
n_scores = cross_val_score(GBR_model_att, X_train_att, y_train_att, scoring='neg_root_mean_squared_error', cv=cv, n_jobs=-1)
# report performance

#Fitting model with trainig data
GBR_model_att.fit(X_train_att,y_train_att)
y_pred_att = GBR_model_att.predict(X_test_att)
mse = mean_squared_error(y_test_att, y_pred_att)
rmse = math.sqrt(mse)

# Saving model to disk
pickle.dump(GBR_model_att, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict(x_att))
print('Training score: {}'.format(GBR_model_att.score(X_train_att, y_train_att)))
print('Test score: {}'.format(GBR_model_att.score(X_test_att, y_test_att)))
print('RMSE: {}'.format(rmse))