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

cols = ['price', 'minimum_nights', 'calculated_host_listings_count'] # The columns you want to search for outliers in

# Calculate quantiles and IQR
Q1 = data[cols].quantile(0.25) # Same as np.percentile but maps (0,1) and not (0,100)
Q3 = data[cols].quantile(0.75)
IQR = Q3 - Q1

# Return a boolean array of the rows with (any) non-outlier column values
condition = ~((data[cols] < (Q1 - 1.5 * IQR)) | (data[cols] > (Q3 + 1.5 * IQR))).any(axis=1)

# Filter our dataframe based on condition
filtered_df = data[condition]
filtered_df_dec =filtered_df.describe(())


count = filtered_df['room_type'].value_counts()
print(count)
avg_price = filtered_df.groupby('room_type')['price'].agg(np.mean)
room_dec = filtered_df.groupby('room_type')['price'].describe()

Entire_home_df = filtered_df[(filtered_df.room_type == "Shared room")]
Entire_home_df_dec = Entire_home_df.describe()


from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
filtered_df['neighbourhood_group'] = labelencoder.fit_transform(filtered_df['neighbourhood_group'])
filtered_df['neighbourhood'] = labelencoder.fit_transform(filtered_df['neighbourhood'])

Entire_home_df = filtered_df[(filtered_df.room_type == "Shared room")]
Entire_home_df_dec = Entire_home_df.describe()

# condition = ~((Entire_home_df[cols] < (Q1 - 1.5 * IQR)) | (Entire_home_df[cols] > (Q3 + 1.5 * IQR))).any(axis=1)
#
# # Filter our dataframe based on condition
# filtered_df_Entire_home_df = Entire_home_df[condition]
# Q1 = Entire_home_df.price.quantile(0.25)
# Q3 = Entire_home_df.price.quantile(0.75)
# IQR = Q3 - Q1
# no_outliers = Entire_home_df[(Q1 - 1.5*IQR < Entire_home_df.price) &  (Entire_home_df.price < Q3 + 1.5*IQR)]
no_outliers  = Entire_home_df [Entire_home_df ['price'].between(39, 80)]
no_outliers_dec = no_outliers.describe()




from sklearn.model_selection import train_test_split
selection = no_outliers [['price', 'neighbourhood_group','neighbourhood','latitude', 'longitude','minimum_nights','availability_365','host_id',
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
              'n_estimators': [100]}

xgb_grid = GridSearchCV(xgb1,
                        parameters,
                        cv = 2,
                        n_jobs = 5,
                        verbose=True)

xgb_grid.fit(X_train,y_train)
y_pred = xgb_grid.predict(X_test)
#
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
