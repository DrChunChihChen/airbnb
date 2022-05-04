# Import all libraries
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np # linear algebra
import matplotlib.pyplot as plt # ploting the data
import seaborn as sns # ploting the data
import math # calculation
data = pd.read_csv("train_data.csv")
data.info()

data_dec = data.describe()
boxplot = data.boxplot(column=['minimum_nights'])
plt.show()
filtered_df = data
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
filtered_df['neighbourhood_group'] = labelencoder.fit_transform(filtered_df['neighbourhood_group'])
filtered_df['neighbourhood'] = labelencoder.fit_transform(filtered_df['neighbourhood'])
filtered_df['room_type'] = labelencoder.fit_transform(filtered_df['room_type'])
filtered_df_dec = filtered_df.describe()
no_outliers  = filtered_df[filtered_df ['price'].between(0, 175)]
no_outliers_dec = no_outliers.describe()

from sklearn.model_selection import train_test_split
selection = no_outliers [['price', 'neighbourhood_group','neighbourhood','latitude', 'longitude','minimum_nights','availability_365','room_type','host_id',
                            'number_of_reviews','calculated_host_listings_count']]
X = selection .drop('price', axis=1).values
Y = selection ['price'].values
X_train_org, X_test_org, y_train, y_test = train_test_split(X, Y, test_size=0.3)
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
s_scaler = StandardScaler()
X_train = s_scaler.fit_transform(X_train_org.astype(np.float))
X_test = s_scaler.transform(X_test_org.astype(np.float))

# X_test_inverse = s_scaler.inverse_transform(X_test)
xgb1 = XGBRegressor()
parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower
              'objective':['reg:linear'],
              'learning_rate': [.01, .03, 0.05, .07], #so called eta value
              'max_depth': [5, 6, 7, 10,15],
              'min_child_weight': [4],
              'subsample': [0.6],
              'colsample_bytree': [0.7],
              'n_estimators': [10]}

xgb_grid = GridSearchCV(xgb1,
                        parameters,
                        cv = 2,
                        n_jobs = 5,
                        verbose=True)

xgb_grid.fit(X_train,y_train)

data_test = pd.read_csv('test_data.csv')
data_test['neighbourhood_group'] = labelencoder.fit_transform(data_test['neighbourhood_group'])
data_test['neighbourhood'] = labelencoder.fit_transform(data_test['neighbourhood'])
data_test['room_type'] = labelencoder.fit_transform(data_test['room_type'])
selection_test = data_test [['neighbourhood_group','neighbourhood','latitude', 'longitude','minimum_nights','availability_365','room_type','host_id',
                            'number_of_reviews','calculated_host_listings_count']].values
selection_test = s_scaler.transform(selection_test.astype(np.float))
y_pred_test = xgb_grid.predict(selection_test)
data_test2 = pd.read_csv('test_data.csv')
ID_column = data_test2["id"]

y_pred_test = pd.Series(y_pred_test,name="price")
results = pd.concat([ID_column, y_pred_test], axis=1)

# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import r2_score
# lr = LinearRegression()
# lr.fit(X_train, y_train)
#
# # Predicting the Test set results
# y_pred = lr.predict(X_test)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
# print('R2 score train:', np.round(r2_score(y_train, lr.predict(X_train), multioutput='variance_weighted'), 2))
# print('R2 score test:', np.round(r2_score(y_test, lr.predict(X_test), multioutput='variance_weighted'), 2))
# print(xgb_grid.feature_importances_)
# # plot
# plt.bar(range(len(xgb_grid.feature_importances_)), xgb_grid.feature_importances_)
# plt.show()
importances = xgb_grid.best_estimator_.feature_importances_


# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import r2_score
# lr = LinearRegression()
# lr.fit(X_train, y_train)
# #
# # # Predicting the Test set results
# y_pred = lr.predict(X_test)
# print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
# print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
# print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
# print('R2 score train:', np.round(r2_score(y_train, lr.predict(X_train), multioutput='variance_weighted'), 2))
# print('R2 score test:', np.round(r2_score(y_test, lr.predict(X_test), multioutput='variance_weighted'), 2))




# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Activation
# from tensorflow.keras.optimizers import Adam
# model = Sequential()
# model.add(Dense(512,activation='relu'))
# model.add(Dense(256,activation='relu'))
# model.add(Dense(128,activation='relu'))
# model.add(Dense(64,activation='relu'))
# model.add(Dense(32,activation='linear'))
# model.add(Dense(1))
# model.compile(optimizer='adam',loss='mse')
# model.fit(x=X_train,y=y_train,
#           validation_data=(X_test,y_test),
#           batch_size=128,epochs=20)
#
# model.summary()
# loss_df = pd.DataFrame(model.history.history)
# loss_df.plot(figsize=(12,8))
#
# y_pred = model.predict(X_test)

from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
print('MSE:', metrics.mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
fig = plt.figure(figsize=(10,5))
plt.scatter(y_test,y_pred)
# Perfect predictions
plt.plot(y_test,y_test,'r')
plt.show()
