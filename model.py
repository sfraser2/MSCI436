#-------------------------------------------------------------------------------------------------------------------------------
import pandas as pd                                                 # Importing package pandas (For Panel Data Analysis)

#-------------------------------------------------------------------------------------------------------------------------------
import numpy as np                                                  # Importing package numpys (For Numerical Python)
#-------------------------------------------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt                                     # Importing pyplot interface to use matplotlib
import seaborn as sns                                               # Importing seaborn library for interactive visualization
%matplotlib inline
#-------------------------------------------------------------------------------------------------------------------------------
import scipy as sp                                                  # Importing library for scientific calculations
#-------------------------------------------------------------------------------------------------------------------------------
import warnings
warnings.filterwarnings('ignore')

pd.options.mode.chained_assignment = None
# for data pipeline --------------------

from sklearn.model_selection import train_test_split
from sklearn.metrics import*
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import make_pipeline

# for prediction (machine learning models) ------------------------

from sklearn.linear_model import*
from sklearn.preprocessing import*
from sklearn.ensemble import*
from sklearn.neighbors import*
from sklearn import svm
from sklearn.naive_bayes import*
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

data = pd.read_csv(filepath_or_buffer = 'https://raw.githubusercontent.com/insaid2018/Term-2/master/Projects/avocado.csv')

print('Data Shape:', data.shape)
nRow, nCol = data.shape
print(f'There are {nRow} rows and {nCol} columns')

data.head()
data.describe()
data.dtypes
data.info()
data.isna().sum()
sns.heatmap(data.isnull());
print('total number of duplicate values : ',sum(data.duplicated()))
data=data.drop(['Unnamed: 0'], axis=1)
data.head()
datam=pd.read_csv(filepath_or_buffer = 'https://raw.githubusercontent.com/insaid2018/Term-2/master/Projects/avocado.csv') # Archieving main dataset
data.select_dtypes('object').columns
sns.set_style("white")

plt.figure(figsize=(12,12))
sns.distplot(data.AveragePrice)
plt.title("Distribution of Average Price",fontsize=12);
import statsmodels.api as sm
sm.qqplot(data.AveragePrice,line="s")
mean = data.AveragePrice.mean()
std = data.AveragePrice.std()
lower, upper = mean-std*2,mean+std*2 # Use 2*std and it will exclude data that is not included in 95% of data
print("Lower Limit : {} Upper Limit : {}".format(lower,upper))
mean = data.AveragePrice.mean()
std = data.AveragePrice.std()
lower, upper = mean-std*2,mean+std*2 # Use 2*std and it will exclude data that is not included in 95% of data
print("Lower Limit : {} Upper Limit : {}".format(lower,upper))
outliers = [x for x in data.AveragePrice if x < lower or x > upper]
print("Outlier values : {}".format(outliers))
df_exclude = data[(data.AveragePrice < upper) | (data.AveragePrice > lower)]
df_exclude.head()
df_exclude.shape
data.shape
quantile = np.quantile(data.AveragePrice,[0.25,0.5,0.75,1]) # Use numpy quantile
IQR = quantile[2] - quantile[0] # Calculate IQR through third quantile - first quantile
upper = 1.5*IQR + quantile[2]
lower = quantile[0] - 1.5*IQR

print("Upper bound : {} Lower bound : {}".format(upper,lower))

outlier = [x for x in data.AveragePrice if x < lower or x>upper]
print("\nOutlier values :\n {}".format(outliers))
df_exclude2 = data[(data.AveragePrice > lower) | (data.AveragePrice < upper)]
df_exclude2

log_data = np.log(data.AveragePrice+1)
sns.set_style("white")
plt.figure(figsize=(8,8))
sns.distplot(log_data);

fig,ax = plt.subplots(1,2,figsize=(10,7))
sm.qqplot(data.AveragePrice,line="s",ax=ax[0])
ax[0].set_title("Before logarithmic")
sm.qqplot(log_data,line="s",ax=ax[1])
ax[1].set_title("After logarithmic");

len(data.region.unique())
data.groupby('region').size()
plt.figure(figsize=(12,5))
plt.title("Distribution Price")
ax = sns.distplot(data["AveragePrice"], color = 'g')
sns.boxplot(y="type", x="AveragePrice", data=data, palette = 'pink');

#Weight distribution of prices
fig, ax = plt.subplots()
fig.set_size_inches(10,5)
sns.violinplot(data.dropna(subset = ['AveragePrice']).AveragePrice);

plt.figure(figsize=(15,15))

plt.title("Avgerage Price of Avocado by Region")

sns.barplot(x="AveragePrice",y="region",data=data)

plt.show()

fig, ax = plt.subplots(1, 2, figsize=(18,5))

regionP = data.groupby('region')['AveragePrice'].mean()

expensive = regionP.sort_values(ascending = False).iloc[:10]
cheap = regionP.sort_values().iloc[:10]

sns.barplot(x='AveragePrice', y='region', data = data, order=expensive.index, ci=None, palette='Greens_r', ax=ax[0])
sns.barplot(x='AveragePrice', y='region', data = data, order=cheap.index, ci=None, palette='Greens_r', ax=ax[1])

plt.show()

cheap = pd.DataFrame(cheap).reset_index()
expensive = pd.DataFrame(expensive).reset_index()

print('the most expensive datacados can be found in {} '.format(list(expensive.iloc[:5,0])))
print('the cheapest datacados can be found in {} '.format(list(cheap.iloc[:5,0])))

fig, ax = plt.subplots(1, 2, figsize=(18,5))

dataStates = data[data['region'] !='TotalUS']

regionV = dataStates.groupby('region')['Total Volume'].sum()

most = regionV.sort_values(ascending = False).iloc[:10]
least = regionV.sort_values().iloc[:10]

sns.barplot(x='Total Volume', y='region', data = dataStates, order=most.index, ci=None, palette='Greens_r', ax=ax[0])
sns.barplot(x='Total Volume', y='region', data = dataStates, order=least.index, ci=None, palette='Greens_r', ax=ax[1])

plt.show()

most = pd.DataFrame(most).reset_index()
least = pd.DataFrame(least).reset_index()

print('States with the the biggest demand are {} '.format(list(most.iloc[:5,0])))
print('States with the least demand are {} '.format(list(least.iloc[:5,0])))

from datetime import datetime
data['Date'] = data['Date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))

data['month'] = data['Date'].dt.month
data['day'] = data['Date'].dt.day
# monday = 0
data['day of week'] = data['Date'].dt.dayofweek
dates = ['year', 'month', 'day', 'day of week']
data[dates]

fig, ax = plt.subplots(2,2, figsize=(20,10))

sns.countplot('year', data=data, ax=ax[0,0], palette='BuGn_r')
sns.countplot('month', data=data, ax=ax[0,1], palette='BuGn_r')
sns.countplot('day', data=data, ax=ax[1,0], palette='BuGn_r')
sns.countplot('day of week', data=data, ax=ax[1,1], palette='BuGn')

plt.show()

data.drop('day of week', axis=1, inplace=True)

fig, ax = plt.subplots(2, 1, figsize=(23,10))

data['year_month'] = data['Date'].dt.to_period('M')
grouped = data.groupby('year_month')[['AveragePrice', 'Total Volume']].mean()

ax[0].plot(grouped.index.astype(str), grouped['AveragePrice'])
ax[0].tick_params(labelrotation=90)
ax[0].set_ylabel('AveragePrice')


ax[1].plot(grouped.index.astype(str), grouped['Total Volume'])
ax[1].tick_params(labelrotation=90)
ax[1].set_ylabel('Total Volume')

plt.show()

fig, ax = plt.subplots(1, 2, figsize=(12,5))

data['quarter'] = data['Date'].dt.quarter


sns.barplot(x='quarter', y='Total Volume', data=data, palette='Greens_r', ci=None, ax=ax[0])
sns.barplot(x='quarter', y='AveragePrice', data=data, palette='Greens_r', ci=None, ax=ax[1])


plt.show()

quarter = data.groupby('quarter')[['Total Volume', 'AveragePrice']].mean()
display(quarter)

data.head()

print(len(data.type.unique()))

data.groupby('type').size()

plt.figure(figsize=(5,7))

plt.title("Avg.Price of Avocados by Type")

sns.barplot(x="type",y="AveragePrice",data= data)

plt.show()

plt.figure(figsize=(18,10))
sns.lineplot(x="month", y="AveragePrice", hue='type', data=data);

data['month'].head()

data['month'] = data['month'].replace({1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 
                                   6: 'June', 7: 'July', 8: 'August', 9: 'September', 10: 'October', 
                                   11: 'November', 12: 'December'})
ax = sns.catplot(x="month", y="AveragePrice", hue="type", 
            kind="box", data=data, height=8.5, linewidth=2.5, aspect=2.8,palette="Set2");

plt.figure(figsize=(18,10))
sns.lineplot(x="month", y="AveragePrice", hue='year',  data=data)
plt.show()

fig, ax = plt.subplots(1, 2, figsize=(14,5))

sns.barplot(x='type', y='AveragePrice', data=data, palette='Set3', ax=ax[0])
sns.barplot(x='type', y='Total Volume', data=data, palette='Set3', ax=ax[1], estimator=sum, ci=None)
plt.show()

display(data.groupby('type')['AveragePrice'].mean())
display(data.groupby('type')['Total Volume'].sum())

from matplotlib.colors import DivergingNorm
fig, ax = plt.subplots(2, 1, figsize=(23,12))
fig.tight_layout(pad=8)


group = data.groupby(['type', 'year_month'])['Total Volume'].sum()

organic = group['organic']
organic = pd.DataFrame(organic)
organic['Total Volume % change'] = np.round(organic['Total Volume'].pct_change() * 100, 2)

conventional = group['conventional']
conventional = pd.DataFrame(conventional)
conventional['Total Volume % change'] = np.round(conventional['Total Volume'].pct_change() * 100, 2)

norm = DivergingNorm(vmin=organic['Total Volume % change'].min(), vcenter=0, vmax=organic['Total Volume % change'].max())
colors = [plt.cm.RdYlGn(norm(c)) for c in organic['Total Volume % change']]
sns.barplot(x=organic.index, y=organic['Total Volume % change'], data=organic, ax=ax[0], palette=colors)

norm = DivergingNorm(vmin=conventional['Total Volume % change'].min(), vcenter=0, vmax=conventional['Total Volume % change'].max())
colors = [plt.cm.RdYlGn(norm(c)) for c in conventional['Total Volume % change']]
sns.barplot(x=conventional.index, y=conventional['Total Volume % change'], data=conventional, ax=ax[1], palette=colors)


ax[0].tick_params(labelrotation=90)
ax[0].set_title('Organic Percentage Change in Sales', fontsize=15)

ax[1].tick_params(labelrotation=90)
ax[1].set_title('Conventional Percentage Change in Sales', fontsize=15)

plt.show()

conventional['Total Volume % change'].mean()
print("The sum of percentage change of Organic is: {}".format(np.around(organic['Total Volume % change'].sum(), 2)))
print("The sum of percentage change of Conventional is: {}".format(np.around(conventional['Total Volume % change'].sum(), 2)))

region_list=list(data.region.unique())
average_price=[]

for i in region_list:
    x=data[data.region==i]
    region_average=sum(x.AveragePrice)/len(x)
    average_price.append(region_average)

data1=pd.DataFrame({'region_list':region_list,'average_price':average_price})
new_index=data1.average_price.sort_values(ascending=False).index.values
sorted_data=data1.reindex(new_index)

plt.figure(figsize=(24,10))
ax=sns.barplot(x=sorted_data.region_list,y=sorted_data.average_price)

plt.xticks(rotation=90)
plt.xlabel('Region')
plt.ylabel('Average Price')
plt.title('Average Price of Avocado According to Region',fontsize=18);

filter1=data.region!='TotalUS'
data1=data[filter1]

region_list=list(data1.region.unique())
average_total_volume=[]

for i in region_list:
    x=data1[data1.region==i]
    average_total_volume.append(sum(x['Total Volume'])/len(x))
data3=pd.DataFrame({'region_list':region_list,'average_total_volume':average_total_volume})

new_index=data3.average_total_volume.sort_values(ascending=False).index.values
sorted_data1=data3.reindex(new_index)

plt.figure(figsize=(22,10))
ax=sns.barplot(x=sorted_data1.region_list,y=sorted_data1.average_total_volume)

plt.xticks(rotation=90)
plt.xlabel('Region')
plt.ylabel('Average of Total Volume')
plt.title('Average of Total Volume According to Region',fontsize=18);

g = sns.factorplot('AveragePrice','region',data=data,
                   hue='year',
                   size=18,
                   aspect=0.7,
                   palette='magma',
                   join=False,
              )

mask = data['type']=='conventional'
g = sns.factorplot('AveragePrice','region',data=data[mask],
                   hue='year',
                   size=13,
                   aspect=0.8,
                   palette='magma',
                   join=False,
              )

numbers = list(data.select_dtypes(['float64', 'int64']).keys())

# removing years
numbers.remove('year')

data[numbers].hist(figsize=(20,10), color='green', edgecolor='white')

plt.show()

display(data[numbers].describe())

data_o = data[data['Total Volume']<50000]
data_o[numbers].hist(figsize=(20,10), color='green', edgecolor='white')

plt.show()

TotalLog = np.log(data['Total Volume'] + 1)
TotalLog.hist(color='green', edgecolor='white');

plt.figure(figsize=(12,6))
sns.heatmap(data.corr(),cmap='coolwarm',annot=True);

from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
dicts = {}

label.fit(data.type.drop_duplicates()) 
dicts['type'] = list(label.classes_)
data.type = label.transform(data.type)

cols = ['AveragePrice','type','year','Total Volume','Total Bags']
cm = np.corrcoef(data[cols].values.T)
sns.set(font_scale = 1.7)
hm = sns.heatmap(cm,cbar = True, annot = True,square = True, fmt = '.2f', annot_kws = {'size':15}, yticklabels = cols, xticklabels = cols)

fig, ax = plt.subplots(1, 2,figsize=(40,15))

data_o = data[data['Total Volume']<50000]

sns.heatmap(data.corr(), vmin=-1, vmax=1, cmap=sns.diverging_palette(20, 220, as_cmap=True), annot=True, ax=ax[0])
ax[0].set_title('With outliers', fontsize=25)

sns.heatmap(data_o.corr(), vmin=-1, vmax=1, cmap=sns.diverging_palette(20, 220, as_cmap=True), annot=True, ax=ax[1])
ax[1].set_title('Without outliers', fontsize=25)

plt.show()

fig, ax = plt.subplots(2, 2, figsize=(20,10))
sns.set(style='whitegrid', rc={"grid.linewidth": 0.1})
sns.set_context("paper", font_scale=0.9)    
sns.scatterplot(x='4046', y='AveragePrice', data=data, hue='type', ax=ax[0,0])
sns.scatterplot(x='Large Bags', y='AveragePrice', data=data_o, hue='type', ax=ax[0,1])
sns.scatterplot(x='month', y='AveragePrice', data=data, hue='type', ax=ax[1,0])
sns.scatterplot(x='month', y='AveragePrice', data=data_o, hue='type', ax=ax[1,1])
# Labels and clean up on the plot                                                                                                                                                                                                                                                                                              
plt.xticks(rotation=90)                                                               
plt.tight_layout()
#plt.savefig('test.pdf', bbox_inches='tight') ;

scaler = Normalizer()
scaler.fit(data[['4046', 'AveragePrice']].values)
data['4046_scaled'] = scaler.transform(data[['4046', 'AveragePrice']].values)[:,0]
data['AveragePrice_scaled'] = scaler.transform(data[['4046', 'AveragePrice']].values)[:,1]

sns.regplot(x='4046_scaled', y='AveragePrice_scaled', data=data, color='g')
plt.show()

# Specifying dependent and independent variables

X = data[['4046', '4225', '4770', 'Small Bags', 'Large Bags', 'XLarge Bags', 'type', 'year', 'region']]
Y = data['AveragePrice']
y=np.log1p(Y)

X.head()
Y.head()

# X_labelled = pd.get_dummies(X[["type","region"]], drop_first = True)
# X_labelled.head()

X = pd.get_dummies(X, prefix=["type","region"], columns=["type","region"], drop_first = True)
X.head()
print(X.columns)

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import mean_squared_error

X_train, X_valid, y_train, y_valid = train_test_split(X, Y, test_size = 0.3, random_state = 99)
X_train.shape, X_valid.shape, y_train.shape, y_valid.shape

lr = LinearRegression()
lr.fit(X_train,y_train)

print("R2 of Linear Regresson:", lr.score(X_train,y_train) )
print("----- Prediction Accuracy-----")
print('MAE: ',metrics.mean_absolute_error(y_valid, lr.predict(X_valid)))
print('MSE: ',metrics.mean_squared_error(y_valid, lr.predict(X_valid)))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_valid, lr.predict(X_valid))))

# Creating a Histogram of Residuals
plt.figure(figsize=(6,4))
sns.distplot(y_valid - lr.predict(X_valid))
plt.title('Distribution of residuals')
plt.show()

plt.scatter(y_valid,lr.predict(X_valid));

from sklearn.svm import SVR

#clf = svm.SVR(kernel = 'linear')
#clf.fit(X_train, y_train)
#confidence = clf.score(X_train, y_train)
#print(k,confidence)
#for k in ['linear','poly','rbf','sigmoid']:
#    print("Running for k as ", k)
#    clf = svm.SVR(kernel=k)
#    clf.fit(X_train, y_train)
#    confidence = clf.score(X_train, y_train)
#    print(k,confidence)

svr = SVR(kernel='rbf', C=1, gamma= 0.5)   # Parameter Tuning to get the best accuracy

svr.fit(X_train,y_train)
print(svr.score(X_train,y_train))

from math import sqrt
# calculate RMSE
error = sqrt(metrics.mean_squared_error(y_valid,svr.predict(X_valid))) 
print('RMSE value of the SVR Model is:', error)
# Creating a Histogram of Residuals
plt.figure(figsize=(6,4))
sns.distplot(y_valid - svr.predict(X_valid))
plt.title('Distribution of residuals')
plt.show()
plt.scatter(y_valid,svr.predict(X_valid));
# Linear Regression RMSE : 
print('RMSE value of the Linear Regr : ',round(np.sqrt(metrics.mean_squared_error(y_valid, lr.predict(X_valid))),4))

# SVR RMSE               : 
print('RMSE value of the SVR Model   : ',round(np.sqrt(metrics.mean_squared_error(y_valid, svr.predict(X_valid))),4))

data=data.drop(['Date'], axis=1)
data_dt=data # for decision tree alogorithm
data=data.drop(['year_month'], axis=1)

X=datam.drop('AveragePrice',1)
y=datam['AveragePrice']

print('shape of X and y respectively :',X.shape,y.shape)
X.head()
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
print('shape of X and y respectively(train) :',X_train.shape,y_train.shape)
print('shape of X and y respectively(test) :',X_test.shape,y_test.shape)
cols=X_train.columns

scaler=LabelEncoder()
for col in X_train.columns:
    if datam[col].dtype=='object':
        X_train[col]=scaler.fit_transform(X_train[col])
        X_test[col]=scaler.transform(X_test[col])
        
X_train.head()
X_train.shape

scaler=VarianceThreshold(0.1)

X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

print('shape of X (train) :',X_train.shape)
print('shape of X (test) :',X_test.shape)

plt.plot(X_train[0]);
scaler=StandardScaler()

X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
print("Type of X_train :",type(X_train))
print("Type of X_test :",type(X_test))

X_train=pd.DataFrame(X_train,columns=cols)
X_train.head()
X_test=pd.DataFrame(X_test,columns=cols)
X_test.head()
print('Type of X_train and X_test :',type(X_train),type(X_test))

actr=[]
acts=[]
lstr=[]
lsts=[]

loss=[]
val_loss=[]

for i in range(2,5):
  clf=make_pipeline(PolynomialFeatures(i),LinearRegression())
  clf.fit(X_train,y_train)
  y_pr=clf.predict(X_test)
  y_x=clf.predict(X_train)
  loss.append(mean_squared_error(y_train,y_x))
  val_loss.append(mean_squared_error(y_test,y_pr))

plt.title('Model Loss')
plt.xlabel('degree')
plt.ylabel('MSE loss')
plt.plot(range(2,5),loss/np.mean(loss),label='train loss')
plt.plot(range(2,5),val_loss/np.mean(val_loss),label='validation loss')
plt.legend()
plt.show()
print('Train loss and validation loss of the polynomial function model :',loss[1],'and',val_loss[1])

clf=make_pipeline(PolynomialFeatures(3),LinearRegression())
clf.fit(X_train,y_train)
print('train accuracy :',clf.score(X_train,y_train))
print('test accuracy :',clf.score(X_test,y_test))

actr.append(clf.score(X_train,y_train))
acts.append(clf.score(X_test,y_test))
lstr.append(loss[1])
lsts.append(val_loss[1])

clf=RandomForestRegressor(random_state=0)
clf.fit(X_train,y_train)
y_tr1=clf.predict(X_train)
y_pr=clf.predict(X_test)

print('train data accuracy :',clf.score(X_train,y_train))
print('test data accuracy :',clf.score(X_test,y_test))
print('loss of train data :',mean_squared_error(y_train,y_tr1))
print('loss of test data :',mean_squared_error(y_test,y_pr))

actr.append(clf.score(X_train,y_train))
acts.append(clf.score(X_test,y_test))
lstr.append(mean_squared_error(y_train,y_tr1))
lsts.append(mean_squared_error(y_test,y_pr))

loss=[]
val_loss=[]
for i in range(1,11):
  clf=Ridge(random_state=0,alpha=i/100.0)
  clf.fit(X_train,y_train)
  y_pr=clf.predict(X_test)
  y_x=clf.predict(X_train)
  loss.append(mean_squared_error(y_train,y_x))
  val_loss.append(mean_squared_error(y_test,y_pr))

plt.title('Model Loss')
plt.xlabel('alpha')
plt.ylabel('MSE loss')
plt.plot(np.arange(1,11,1)/100,loss/np.mean(loss),label='train loss')
plt.plot(np.arange(1,11,1)/100,val_loss/np.mean(val_loss),label='validation loss')
plt.legend()
plt.show()

clf=Ridge(random_state=0,alpha=0.01)
clf.fit(X_train,y_train)
y_tr1=clf.predict(X_train)
y_pr=clf.predict(X_test)

print('train data accuracy :',clf.score(X_train,y_train))
print('test data accuracy :',clf.score(X_test,y_test))
print('loss of train data :',mean_squared_error(y_train,y_tr1))
print('loss of test data :',mean_squared_error(y_test,y_pr))
actr.append(clf.score(X_train,y_train))
acts.append(clf.score(X_test,y_test))
lstr.append(mean_squared_error(y_train,y_tr1))
lsts.append(mean_squared_error(y_test,y_pr))

loss=[]
val_loss=[]
for i in range(1,11):
  clf=Lasso(random_state=0,alpha=i/100.0)
  clf.fit(X_train,y_train)
  y_pr=clf.predict(X_test)
  y_x=clf.predict(X_train)
  loss.append(mean_squared_error(y_train,y_x))
  val_loss.append(mean_squared_error(y_test,y_pr))
  
plt.title('Model Loss')
plt.xlabel('alpha')
plt.ylabel('MSE loss')
plt.plot(np.arange(1,11,1)/100,loss/np.mean(loss),label='train loss')
plt.plot(np.arange(1,11,1)/100,val_loss/np.mean(val_loss),label='validation loss')
plt.legend()
plt.show()

clf=Lasso(random_state=0,alpha=0.01)
clf.fit(X_train,y_train)
y_tr1=clf.predict(X_train)
y_pr=clf.predict(X_test)

print('train data accuracy :',clf.score(X_train,y_train))
print('test data accuracy :',clf.score(X_test,y_test))
print('loss of train data :',mean_squared_error(y_train,y_tr1))
print('loss of test data :',mean_squared_error(y_test,y_pr))

actr.append(clf.score(X_train,y_train))
acts.append(clf.score(X_test,y_test))
lstr.append(mean_squared_error(y_train,y_tr1))
lsts.append(mean_squared_error(y_test,y_pr))

clf=BayesianRidge()
clf.fit(X_train,y_train)
y_tr1=clf.predict(X_train)
y_pr=clf.predict(X_test)

print('train data accuracy :',clf.score(X_train,y_train))
print('test data accuracy :',clf.score(X_test,y_test))
print('loss of train data :',mean_squared_error(y_train,y_tr1))
print('loss of test data :',mean_squared_error(y_test,y_pr))

actr.append(clf.score(X_train,y_train))
acts.append(clf.score(X_test,y_test))
lstr.append(mean_squared_error(y_train,y_tr1))
lsts.append(mean_squared_error(y_test,y_pr))

models=['Polynomial','Random Forest','Ridge','Lasso','Bayesian Ridge']
plt.title('Model Accuracy')
plt.plot(models,actr,label='train data')
plt.plot(models,acts,label='validation data')
plt.legend()
plt.show()

plt.title('Model Loss')
plt.plot(models,lstr,label='train data')
plt.plot(models,lsts,label='validation data')
plt.legend()
plt.show()

from sklearn.model_selection import train_test_split

trainflights, testflights, ytrain, ytest = train_test_split(data, y, train_size=0.7,test_size=0.3, random_state=0)
s = (trainflights.dtypes == 'object')
object_cols = list(s[s].index)

n = (trainflights.dtypes == ('float64','int64'))
numerical_cols = list(n[n].index)
#checking the columns containing categorical columns:
print(object_cols)

#using One Hot Encoder to make the categorical columns usable

oneHot = OneHotEncoder(handle_unknown = 'ignore', sparse=False)
oneHottrain = pd.DataFrame(oneHot.fit_transform(trainflights[object_cols]))
oneHottest = pd.DataFrame(oneHot.transform(testflights[object_cols]))

#reattaching index since OneHotEncoder removes them:
oneHottrain.index = trainflights.index
oneHottest.index = testflights.index 

#dropping the old categorical columns:
cattraincol = trainflights.drop(object_cols, axis=1)
cattestcol = testflights.drop(object_cols, axis=1)

#concatenating the new columns:
trainflights = pd.concat([cattraincol, oneHottrain], axis=1)
testflights = pd.concat([cattestcol, oneHottest], axis=1)

#scaling the values

trainf = trainflights.values
testf = testflights.values

minmax = MinMaxScaler()

trainflights = minmax.fit_transform(trainf)
testflights = minmax.transform(testf)

#defining a way to find Mean Absolute Percentage Error:
def PercentError(preds, ytest):
  error = abs(preds - ytest)

  errorp = np.mean(100 - 100*(error/ytest))

  print('the accuracy is:', errorp)
  #implementing the algo:
model = RandomForestRegressor(n_estimators=100, random_state=0, verbose=1)

#fitting the data to random forest regressor:
model.fit(trainflights, ytrain)

#predicting the test dataset:
preds = model.predict(testflights)
PercentError(preds, ytest)

#using linear regression:
LinearModel = LinearRegression()
LinearModel.fit(trainflights, ytrain)

#predicting on the test dataset:
LinearPredictions = LinearModel.predict(testflights)
PercentError(LinearPredictions, ytest)

df=data[["year","Small Bags","Large Bags", "AveragePrice"]]

df = df.sample(n=50,replace=True)
#df=df.head(50)
df.tail()

y=df.iloc[:,1].values
x=df.iloc[:,-1].values

x=x.reshape(len(x),1)
y=y.reshape(len(y),1)

from sklearn.tree import DecisionTreeRegressor
regressor=DecisionTreeRegressor(random_state=0)
regressor.fit(x,y)
DecisionTreeRegressor(random_state=0)
## Predicting a new result: What is the price for the 
regressor.predict([[8042.21]])
regressor.predict([[8000]])
regressor.predict([[18000]])

X_grid = np.arange(min(x), max(x), 0.001)
X_grid = X_grid.reshape((len(X_grid), 1))

plt.scatter(x,y, color="red")
plt.plot(X_grid, regressor.predict(X_grid), color="blue")

plt.title("Decision Tree Regression for Avocado Prices")
plt.xlabel("Avocado Small Bags")
plt.ylabel("Price")
plt.show()
