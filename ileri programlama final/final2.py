import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")
data=pd.read_csv('california_housing.csv')
data.dropna(inplace=True)
print(data.info())
data = data.join(pd.get_dummies(data.ocean_proximity)).drop(['ocean_proximity'],axis=1)


x=data.drop(['median_house_value'],axis=1)
y=data['median_house_value']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
train_data=x_train.join(y_train)


# train_data_wo_proximity=train_data.drop(['ocean_proximity'],axis=1)
# train_data_wo_proximity.corr()
plt.figure(figsize=(15,8))
sns.heatmap(train_data.corr(),annot=True,cmap='RdYlGn')

train_data['total_rooms']=np.log(train_data['total_rooms'])
train_data['total_bedrooms']=np.log(train_data['total_bedrooms'])
train_data['population']=np.log(train_data['population'])
train_data['households']=np.log(train_data['households'])

# train_data.ocean_proximity.value_counts()
# train_data = train_data.join(pd.get_dummies(train_data.ocean_proximity)).drop(['ocean_proximity'],axis=1)
plt.figure(figsize=(15,8))
sns.heatmap(train_data.corr(),annot=True,cmap='YlGnBu')
sns.scatterplot(x='latitude',y='longitude',data=train_data,hue='median_house_value',palette="coolwarm")

# train_data['bedroom_ratio']=train_data['total_bedrooms']/train_data['total_rooms']
# train_data['household_rooms']=train_data['total_rooms']/train_data['households']

# x_train,y_train=train_data.drop(['median_house_value'],axis=1),train_data['median_house_value']
# scaler=StandardScaler()
# x_train_s=scaler.fit_transform(x_train)

test_data=x_test.join(y_test)
test_data['total_rooms']=np.log(test_data['total_rooms'])
test_data['total_bedrooms']=np.log(test_data['total_bedrooms'])
test_data['population']=np.log(test_data['population'])
test_data['households']=np.log(test_data['households'])

# reg=LinearRegression()
# reg.fit(x_train,y_train)
# reg.score(x_test,y_test)


forest=RandomForestRegressor()
forest.fit(x_train,y_train)
forest.score(x_test,y_test)
y_pred = forest.predict(x_test)
y_comp=pd.DataFrame(y_test)
y_comp1=y_comp.join(pd.DataFrame({'prediction':y_pred}).set_index(y_test.index))
print(y_comp1)

# knn=KNeighborsRegressor()
# knn.fit(x_train,y_train)
# knn.score(x_test,y_test)

# mlp=MLPRegressor()
# mlp.fit(x_train,y_train)
# mlp.score(x_test,y_test)

# param_grid = { 
#     "n_estimators": [200, 500],
#     "max_features": ["auto", "sqrt", "log2"],
# }
# grid = GridSearchCV(forest, param_grid, cv=5,scoring='neg_mean_squared_error',return_train_score=True)
# grid.fit(x_train, y_train)  


# param_grid = { 
#     "n_estimators": [100, 500],
#     "min_samples_split":[2,4,6,8],
#     "max_depth":[None,4,8]
# }
# grid = GridSearchCV(forest, param_grid, cv=5,scoring='neg_mean_squared_error',return_train_score=True)
# grid.fit(x_train, y_train)  


