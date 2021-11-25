import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
# import seaborn as sns
import math
import scipy.stats as st

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer


# # read data to dataframe
# df_crime = pd.read_csv("nt_crime_statistics_jun_2021.csv") 
df_temp = pd.read_csv("IDCJAC0002_014015_Data12.xls")
df_unemp = pd.read_csv("Unemployment rate-SA4 Time Series - September 2021.csv")
df_alcoh = pd.read_csv('alcohol-related-assaults-monthly-data.csv')


# ~~~~ Wrangle Unemployment training Dataset ~~~~
# split month and year
df_unemp[["month", "year"]] = df_unemp["Date"].str.split("-", expand = True)
df_unemp["year"] =df_unemp["year"].astype(int)
# print(df_unemp)

df_unemp_new = df_unemp[(df_unemp["State/Territory"]=="NT") & (df_unemp["year"].between(8, 17))]
df_unemp_new = df_unemp_new.drop(["State/Territory", "Date", "Participation Rate (15+)    "], axis=1)
df_unemp_new["year"] = df_unemp_new["year"] + 2000
# print(df_unemp_new)

###~~~unemployment test dataset range 2018-2021 Jul~~~

df_unemp_test_new = df_unemp[(df_unemp["State/Territory"]=="NT") & (df_unemp["year"].between(18, 21))]
df_unemp_test_new = df_unemp_test_new.drop(["State/Territory", "Date", "Participation Rate (15+)    "], axis=1)
df_unemp_test_new["year"] = df_unemp_test_new["year"] + 2000
# print(df_unemp_test_new)


# ~~~~ Wrangle Temperature training Dataset ~~~~2008-2017
# found that temperature data frame month value is by columns not rows

df_temp_new = df_temp.melt(id_vars=["Product code", "Station Number", "Year"], var_name="month", value_name="temp")
df_temp_new = df_temp_new[(df_temp_new["Year"].between(2008, 2017))]
df_temp_new = df_temp_new.drop(["Product code","Station Number"], axis=1)
df_temp_new = df_temp_new.rename(columns={"Year": "year"})
# print(df_temp_new)

#~~~~~~~temperature test dataset~~~~~~~~2018-2021 Jul#

df_temp_test_new = df_temp.melt(id_vars=["Product code", "Station Number", "Year"], var_name="month", value_name="temp")
df_temp_test_new = df_temp_test_new[(df_temp_test_new["Year"].between(2018, 2021))]
df_temp_test_new = df_temp_test_new.drop(["Product code","Station Number"], axis=1)
df_temp_test_new = df_temp_test_new.rename(columns={"Year": "year"})
# print(df_temp_test_new)


#~~~~~training crime dataset 2008-2017~~~~~
df_cr=pd.read_csv('train_Jan2008-Dec2017.csv')
df_cr_new=df_cr.groupby(["Year", "Month number"], as_index=False).sum()
df_cr_grouped = df_cr_new.replace({"Month number": {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 
                                            6: "Jun", 7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}})

df_cr_new = df_cr_grouped.rename(columns={"Year": "year","Month number": "month"})

# print(df_cr_new)


#~~~~test crime dataset 2018-2021 Jul~~~~
df_cr_test=pd.read_csv('test_Jan2018-Jul2021.csv')
df_cr_test_new=df_cr_test.groupby(["Year", "Month number"], as_index=False).sum()
df_cr_test_grouped = df_cr_test_new.replace({"Month number": {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 
                                            6: "Jun", 7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}})

df_cr_test_new = df_cr_test_grouped.rename(columns={"Year": "year","Month number": "month"})

# print(df_cr_test_new)


#~~~~~~~Wrangle Alochol-related offences training Dataset 2008-2017~~~~~~~
df_alcoh_new = df_alcoh[(df_alcoh['Year'].between(2008,2017))]
df_alcoh_new=df_alcoh_new.drop(['Alcohol-related assault offences involving DV','Alcohol-related assault offences not involving DV'],axis=1)
df_alcoh_new['Total Alcohol-related assault offences']=df_alcoh_new['Total Alcohol-related assault offences'].astype(int)
df_alcoh_grouped = df_alcoh_new.groupby(['Year','Month'],as_index=False).sum()

df_alcoh_new1=df_alcoh_grouped.rename(columns={'Year':'year','Month':'month'})
# print(df_alcoh_new1)

#~~~~~~~~Alochol-related offences test Dataset~~~2018-2021 Jul~~~~~

df_alcoh_test_new = df_alcoh[df_alcoh['Year'].between(2018,2021)&(df_alcoh['Region'].isin(['Darwin','Palmerston','NT Balance']))]
df_alcoh_test_new=df_alcoh_test_new.drop(['Alcohol-related assault offences involving DV','Alcohol-related assault offences not involving DV'],axis=1)

df_alcoh_test_new1=df_alcoh_test_new.drop(df_alcoh_test_new[(df_alcoh_test_new['Year']==2021)&(df_alcoh_test_new['Month'].isin(['Aug','Sep','Oct','Nov','Dec']))].index)

# print(df_alcoh_test_new1)

df_alcoh_test_new1['Total Alcohol-related assault offences']=df_alcoh_test_new1['Total Alcohol-related assault offences'].astype(int)
df_alcoh_test_grouped = df_alcoh_test_new1.groupby(['Year','Month'],as_index=False).sum()


df_alcoh_test_new1=df_alcoh_test_grouped.rename(columns={'Year':'year','Month':'month'})
# print(df_alcoh_test_new1)


# ~~~~ Merge ~~~~training dataset~~~
df_data1 = df_unemp_new.merge(df_cr_new, on=["month","year"])
df_data2 = df_data1.merge(df_temp_new, on=["month","year"])

df_data3=df_data2.merge(df_alcoh_new1,on=['year','month'])
df_data3=df_data3[['Employment Rate (15-64)   ','Unemployment Rate (15+)  ','temp','Total Alcohol-related assault offences','Number of offences']]
# print(df_data3)


#~~~~~~Merge ~~~~test dataset~~~~

df_data_test1 = df_unemp_test_new.merge(df_cr_test_new, on=["month","year"])
df_data_test2 = df_data_test1.merge(df_temp_test_new, on=["month","year"])

df_data_test3=df_data_test2.merge(df_alcoh_test_new1,on=['year','month'])
df_data_test3=df_data_test3[['Employment Rate (15-64)   ','Unemployment Rate (15+)  ','temp','Total Alcohol-related assault offences','Number of offences']]
# print(df_data_test3)


#~~~~Separate explanatory variables (x_train) from the response variable (y_train)~~~
X_train = df_data3.iloc[:,:-1].values
y_train = df_data3.iloc[:,-1].values
# print(X_train)
# print(y_train)

#~~~~Separate explanatory variables (x_test) from the response variable (y_test)~~~~
X_test = df_data_test3.iloc[:,:-1].values
y_test = df_data_test3.iloc[:,-1].values
# print(X_test)
# print(y_test)


# Build a linear regression model
model = LinearRegression()

# Train (fit) the linear regression model using the training set
model.fit(X_train, y_train)

# Printing the intercept and coefficient learned by the linear regression model
print("Intercept: ", model.intercept_)
print("Coefficient: ", model.coef_)

# # Use linear regression to predict the values of (y) in the test set
# based on the values of x in the test set
y_pred = model.predict(X_test)

# Put the predicted values of (y) next to the actual values of (y)
df_testdata3_pred = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
print(df_testdata3_pred)

### Compute the standard performance metrics of the linear regression:###

# Mean Absolute Error
mae = metrics.mean_absolute_error(y_test, y_pred)
# Mean Squared Error
mse = metrics.mean_squared_error(y_test, y_pred)
# Root Mean Square Error
rmse =  math.sqrt(metrics.mean_squared_error(y_test, y_pred))
# Normalised Root Mean Square Error
y_max = y_test.max()
y_min = y_test.min()
r = y_max - y_min
rmse_norm = rmse / r

print("Linear Regression metrics:")
print("MAE: ", mae)
print("MSE: ", mse)
print("RMSE: ", rmse)
print("RMSE (Normalised): ", rmse_norm)


