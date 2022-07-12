# %% [markdown] {"id":"9ku0h8OsoHJz"}
# <p align="center"><img src="https://github.com/insaid2018/Term-1/blob/master/Images/INSAID_Full%20Logo.png?raw=true" width="260" height="110" /></p>

# %% [markdown] {"id":"-7Q1C_q31Ejc"}
# <center><img src=  "https://images-na.ssl-images-amazon.com/images/I/71qrXkl0QJL._AC_SY355_.jpg">

# %% [markdown] {"id":"DYXpnBWS1Ejd"}
# <center><a href="mailto:pati.simayan@gmail.com" target="_blank" rel="noopener">Simayan Pati</a>

# %% [markdown] {"id":"GvPYamSK1Ejd"}
# This notebook is submitted for the fulfilment of **GCDAI - Term III**  curriculum [Nov 2020 batch] from **INSAID**. 
# 
#   - An **exploratory data analysis & data visualisation along with implementation of machine learning** been used covering the following topics
#   
#     - Python basics
#     - Numerical computing with Numpy array operations 
#     - Analyzing tabular data with Pandas
#     - Data visualization with Matplotlib and Seaborn 
#     - Fundamental of Machine Learning for prediction and classification using sklearn, SciPy & Skit-Learn
#   - It contains a hypothetical case study for **Hormel Foods,USA** in context to under the pricing phemonon for Hass Avocados using the [available dataset](https://raw.githubusercontent.com/insaid2018/Term-2/master/Projects/avocado.csv).
#   - This data was downloaded from the Hass Avocado Board website in May of 2018 & compiled into a single CSV. Here's [how the Hass Avocado Board describes the data on their website](https://hassavocadoboard.com/):
# 
# > <font color='DarkMagenta'>*The table below represents weekly 2018 retail scan data for National retail volume (units) and price. Retail scan data comes directly from retailers’ cash registers based on actual retail sales of Hass avocados. Starting in 2013, the table below reflects an expanded, multi-outlet retail data set. Multi-outlet reporting includes an aggregation of the following channels: grocery, mass, club, drug, dollar and military. The Average Price (of avocados) in the table reflects a per unit (per avocado) cost, even when multiple units (avocados) are sold in bags. The Product Lookup codes (PLU’s) in the table are only for Hass avocados. Other varieties of avocados (e.g. greenskins) are not included in this table.*
# 
#   - Many thanks to the Hass Avocado Board for sharing this [data](http://www.hassavocadoboard.com/retail/volume-and-price-data)

# %% [markdown]
# > The **Hass avocado** is a cultivar of avocado with dark green–colored, bumpy skin. It was first grown and sold by Southern California mail carrier and amateur horticulturist Rudolph Hass, who also gave it his name. The Hass avocado is a large-sized fruit weighing 200 to 300 grams. When ripe, the skin becomes a dark purplish-black and yields to gentle pressure. When ready to serve, it becomes white-green in the middle part of the inner fruit. Owing to its taste, size, shelf-life, high growing yield and in some areas, year-round harvesting, the Hass cultivar is the most commercially popular avocado worldwide. In the United States it accounts for more than 80% of the avocado crop, 95% of the California crop and is the most widely grown avocado in New Zealand.

# %% [markdown] {"id":"mA6VC1r8DH8g"}
# <a name = Section0></a>
# ---
# # **Table of Contents**
# ---
# 
# 1. [**Introduction**](#Section1)<br>
# 2. [**Problem Statement**](#Section2)<br>
# 3. [**Installing & Importing Libraries**](#Section3)<br>
# 4. [**Data Acquisition & Description**](#Section4)<br>
# 5. [**Data Pre-Processing**](#Section5)<br>
# 6. [**Exploratory Data Analysis**](#Section6)<br>
# 9. [**Summarization**](#Section9)</br>
#   9.1 [**Conclusion**](#Section91)</br>
#   9.2 [**Actionable Insights**](#Section91)</br>

# %% [markdown] {"id":"y7ahy5zG1Eje"}
# ---
# <a name = Section1></a>
# # **1. Introduction**                            
# ---
# [Go back to Index](#Section0)<br>   
# <font color= 'OliveDrab'>**In which cities can millennials have their avocado toast AND buy a home?** It is a well known fact that Millenials LOVE Avocado Toast. It's also a well known fact that all Millennials live in their parents' basements.
# Clearly, they aren't buying home because they are buying too much Avocado Toast! But maybe there's hope… if a Millennial could find a city with cheap avocados, they could live out the Millennial American Dream.<font>
# 
# * Avocados are the darling of the produce section. They’re the go-to ingredient for guacamole dips at parties. And they're also turning up in everything from salads and wraps to smoothies and even brownies
# 
# * Given the rise of Avocadopocalypse in 2017 **Hormel Foods Corporation**, makers of the **WHOLLY GUACAMOLE®** brand, America's #1 selling refrigerated guacamole is interested to understand the volatility & price dynamics basis available data.

# %% [markdown] {"id":"SUOuq1Y22psu"}
#     WHOLLY® GUACAMOLE Classic Guacamole: 
#     Hass Avocados, Distilled Vinegar, Contains 2% Or Less of Water, Jalapeño Peppers, Salt, Dehydrated Onion, Granulated Garlic.

# %% [markdown] {"id":"Iq1uAduF1Ejf"}
# ---
# <a name = Section2></a>
# # **2. Problem Statement**
# ---
# [Go back to Index](#Section0)<br>   
# 
# • Hormel Foods Corps avocados are sourced from over 1000 growers owning over 65,000 acres across California, Mexico, Chile, and Peru.
# 
# • With generations of experience growing, packing, and shipping avocados, they have a deep understanding of the avocado industry.
# 
# • Their aim is to source quality fruit that’s sustainably grown and handled in the most efficient, shortest supply route possible.
# 
# • They want to increase their supply throughout the United States and need to make sure that they are selling their products at the best possible price.
# 
# • Avocado prices have rocketed in recent years by up to 129%, with the average national price in the US of a single Hass avocado reaching $2.10 in 2019, almost doubling in just one year.
# 
# • Due to this uncertainty in the prices, the company is not able to sell their produce at the optimal price.
# 
# • **Task is to predict the optimal price of the avocado using the previous sales data of avocado according to different regions**.

# %% [markdown] {"id":"MzDL9wBh1Ejh"}
# **Steps for our Forecasting Project:**
# 1. Determine what is the problem: In this case we want to have accurate forecast of Avocado prices.
# 2. Gathering Information: Understand what was the process that was used to gather the information and if the information is sufficient to have effective predictive models.
# 3. Implementing Exploratory Analysis: Determine if there are any sort of patterns in our data before going into building the models.
# 4. Choosing predictive models: This is the phase where we decide which model is the most appropriate to make our forecasting most effective.
# 5. Testing our model: Analyze if our model is effective enough to make effective predictions.

# %% [markdown] {"id":"SrpROjt2G_Jx"}
# ---
# <a id = Section3></a>
# # **3. Installing & Importing Libraries**
# ---
# [Go back to Index](#Section0)<br>   
# - This section is emphasised on installing and importing the necessary libraries that will be required.

# %% [code] {"id":"RjFtVFlU1Ejh","jupyter":{"outputs_hidden":false}}
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

# %% [code] {"id":"ZVm5r-Zx1Eji","jupyter":{"outputs_hidden":false}}
import warnings
warnings.filterwarnings('ignore')

pd.options.mode.chained_assignment = None

# %% [code] {"id":"nDkVCDYz1Eji","jupyter":{"outputs_hidden":false}}
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

# %% [markdown] {"id":"7_OBzOaoHGcA"}
# ---
# <a name = Section4></a>
# # **4. Data Acquisition & Description**
# ---
# [Go back to Index](#Section0)<br>

# %% [code] {"id":"UxScFLFC1Eji","jupyter":{"outputs_hidden":false}}
data = pd.read_csv(filepath_or_buffer = 'https://raw.githubusercontent.com/insaid2018/Term-2/master/Projects/avocado.csv')

# %% [code] {"id":"kpOaMbg21Ejj","outputId":"e73d7e0c-3673-49cc-d6ef-3b1cc7cf1255","jupyter":{"outputs_hidden":false}}
print('Data Shape:', data.shape)
nRow, nCol = data.shape
print(f'There are {nRow} rows and {nCol} columns')

# %% [code] {"id":"38r4JO1b1Ejj","outputId":"9934a9ed-0023-4441-c14d-4213f49d1ed3","jupyter":{"outputs_hidden":false}}
data.head()

# %% [code] {"id":"YC7k9k6e1Ejj","outputId":"3c461e4d-52c7-465d-aba4-535f547c00f5","jupyter":{"outputs_hidden":false}}
data.describe()

# %% [code] {"id":"sC3aSmIL1Ejj","outputId":"0d0bbea4-9b5c-4de5-cd0a-5312aa2e60d2","jupyter":{"outputs_hidden":false}}
data.dtypes

# %% [markdown] {"id":"PrGn04F31Ejg"}
# **To predict the future price of avocados depending on some variables / features we have such as**
# 
# - Date - The date of the observation
# - AveragePrice - the average price of a single avocado
# - Total Volume - Total number of avocados sold (small Hass + Large Hass + XLarge Hass + Total Bags)
# - 4046 - Total number of avocados with PLU 4046 sold
# - 4225 - Total number of avocados with PLU 4225 sold
# - 4770 - Total number of avocados with PLU 4770 sold
# - Total Bags = Small Bags + Large Bags + XLarge Bags
# - type - conventional or organic
# - year - the year
# - region - the city or region of the observation

# %% [markdown] {"id":"SNnIcfCxN2hP"}
# ---
# <a name = Section5></a>
# # **5. Data Pre-Processing**
# ---
# [Go back to Index](#Section0)<br>   
# - This section is emphasised on performing data manipulation over unstructured data for further processing and analysis.
# 
# - To modify unstructured data to strucuted data you need to verify and manipulate the integrity of the data by:
#   - Handling missing data,
# 
#   - Handling redundant data,
# 
#   - Handling inconsistent data,
# 
#   - Handling outliers,
# 
#   - Handling typos

# %% [markdown] {"id":"FePapBKG1Ejj"}
# #### Null & Duplicate Entry check:

# %% [code] {"id":"veUKGxPp1Ejk","outputId":"ee233593-485c-4713-8394-5c9879e015c5","jupyter":{"outputs_hidden":false}}
data.info()

# %% [code] {"id":"o2bRwUst1Ejk","outputId":"a820c5eb-d9e2-4e06-91bb-589c778d4840","jupyter":{"outputs_hidden":false}}
data.isna().sum()

# %% [code] {"id":"R-tYsfiN1Ejk","outputId":"42faf289-d0ec-4172-af1f-47d806134a2c","jupyter":{"outputs_hidden":false}}
sns.heatmap(data.isnull());

# %% [code] {"id":"D26oOnr91Ejk","outputId":"bc3d9206-7d36-47ab-a219-0365c550a66b","jupyter":{"outputs_hidden":false}}
print('total number of duplicate values : ',sum(data.duplicated()))

# %% [markdown] {"id":"H--_bVgN1Ejk"}
# **We don't have any null or duplicate value. Lets continue with the descriptive analysis and further. The first column gives reduntant index data, so lets drop it**

# %% [code] {"id":"vhR9ckXV1Ejl","jupyter":{"outputs_hidden":false}}
data=data.drop(['Unnamed: 0'], axis=1)

# %% [code] {"id":"bvwpVaZu1Ejl","outputId":"bab9e99c-ac25-4690-de2d-74fd5f07a917","jupyter":{"outputs_hidden":false}}
data.head()

# %% [code] {"jupyter":{"outputs_hidden":false}}
datam=pd.read_csv(filepath_or_buffer = 'https://raw.githubusercontent.com/insaid2018/Term-2/master/Projects/avocado.csv') # Archieving main dataset

# %% [markdown] {"id":"r3DC-G4y1Ejl"}
# #### String Check

# %% [code] {"id":"HNIuLHBV1Ejl","outputId":"7c2f9ff7-d220-47db-86b0-a184b010b8b3","jupyter":{"outputs_hidden":false}}
data.select_dtypes('object').columns

# %% [markdown] {"id":"WWgxHcC01Ejm"}
# ## Outlier Detection

# %% [markdown] {"id":"ZYQdXyZ-1Ejm"}
# We can use **Standard Deviation** or **Quantile** to detect if any outliers

# %% [code] {"id":"V1NUFSyO1Ejm","outputId":"de496218-915e-4b7a-c20e-5181f68ea517","jupyter":{"outputs_hidden":false}}
sns.set_style("white")

plt.figure(figsize=(12,12))
sns.distplot(data.AveragePrice)
plt.title("Distribution of Average Price",fontsize=12);

# %% [markdown] {"id":"hjk81H161Ejm"}
# #### <font color="blue">Looks like positive skewed slightly

# %% [code] {"id":"KpBN7gkr1Ejm","outputId":"f9f76e76-dbee-48f5-e729-b47171a92957","jupyter":{"outputs_hidden":false}}
import statsmodels.api as sm
sm.qqplot(data.AveragePrice,line="s")

# %% [markdown] {"id":"ZTKiTfEP1Ejm"}
# >- 1 std from mean : 68% of data included
# - 2 std from mean : 95% of data included
# - 3 std from mean : 99.7% of data included

# %% [code] {"id":"sinIk8Pw1Ejn","outputId":"8134be0e-b030-4bde-fa82-1cdf0bd064b9","jupyter":{"outputs_hidden":false}}
mean = data.AveragePrice.mean()
std = data.AveragePrice.std()
lower, upper = mean-std*2,mean+std*2 # Use 2*std and it will exclude data that is not included in 95% of data
print("Lower Limit : {} Upper Limit : {}".format(lower,upper))

# %% [code] {"id":"e5xIbpWo1Ejn","outputId":"db37aa2a-0c3e-489a-dd0c-b9c6977f9dad","jupyter":{"outputs_hidden":false}}
outliers = [x for x in data.AveragePrice if x < lower or x > upper]
print("Outlier values : {}".format(outliers))

# %% [markdown] {"id":"5cm0pMmr1Ejn"}
# #### There is some data that is not included within 95% of data

# %% [code] {"id":"C4xOBWBP1Ejn","jupyter":{"outputs_hidden":false}}
df_exclude = data[(data.AveragePrice < upper) | (data.AveragePrice > lower)]

# %% [code] {"id":"6NCa5eqU1Ejn","outputId":"25c8dd20-ca39-4840-8ea9-09cab7dc7a74","jupyter":{"outputs_hidden":false}}
df_exclude.head()

# %% [code] {"id":"kHb-XcZk1Ejn","outputId":"006ce4c3-2d3f-49a9-d0dc-8f2a86e143d3","jupyter":{"outputs_hidden":false}}
df_exclude.shape

# %% [code] {"id":"SFm70-cY1Ejn","outputId":"78b4fe8d-2b02-44ee-ce44-660732ec12ce","jupyter":{"outputs_hidden":false}}
data.shape

# %% [markdown] {"id":"c2Zxbocs1Ejo"}
# >- Q1 : Data that is located in 25% of total data
# - Q2 : Median value of data
# - Q3 : Data that is located in 75% of total data
# 
# $IQR = Q3 - Q1$
# 
# **Outlier:**
# 
# $Upper bound : 1.5*IQR + Q3$
# 
# $Lower bound : 1.5*IQR - Q1$

# %% [code] {"id":"wSSNUUL81Ejo","outputId":"296866e5-c7e5-41b9-ab74-08b31eb276dc","jupyter":{"outputs_hidden":false}}
quantile = np.quantile(data.AveragePrice,[0.25,0.5,0.75,1]) # Use numpy quantile
IQR = quantile[2] - quantile[0] # Calculate IQR through third quantile - first quantile
upper = 1.5*IQR + quantile[2]
lower = quantile[0] - 1.5*IQR

print("Upper bound : {} Lower bound : {}".format(upper,lower))

outlier = [x for x in data.AveragePrice if x < lower or x>upper]
print("\nOutlier values :\n {}".format(outliers))

# %% [code] {"id":"5c-vjmBH1Ejo","outputId":"fb456869-c8ee-4639-9aa1-e13f42ba736e","jupyter":{"outputs_hidden":false}}
df_exclude2 = data[(data.AveragePrice > lower) | (data.AveragePrice < upper)]
df_exclude2

# %% [markdown] {"id":"BYskfFHg1Ejo"}
# ### Data Normalization
# 
# It is important to check whether data follow normal distribution before we do modeling
# There is one easy way to do normalization, use logarithmic scale

# %% [code] {"id":"lWFEKbRv1Ejo","outputId":"9694a35b-ce8c-4977-c06e-17f908eeb3e9","jupyter":{"outputs_hidden":false}}
log_data = np.log(data.AveragePrice+1)
sns.set_style("white")
plt.figure(figsize=(8,8))
sns.distplot(log_data);

# %% [code] {"id":"EbrUzBya1Ejo","outputId":"523df749-d32f-4da8-d576-ca9846b4205e","jupyter":{"outputs_hidden":false}}
fig,ax = plt.subplots(1,2,figsize=(10,7))
sm.qqplot(data.AveragePrice,line="s",ax=ax[0])
ax[0].set_title("Before logarithmic")
sm.qqplot(log_data,line="s",ax=ax[1])
ax[1].set_title("After logarithmic");

# %% [markdown] {"id":"jXuKUX8K1Ejo"}
# #### It looks much closer to normal distribution after doing logarithmic

# %% [markdown] {"id":"S7omk2LK1Ejp"}
# ---
# <a name = Section6></a>
# # **6. EDA & Class Imbalance Check**                            
# ---
# [Go back to Index](#Section0)<br>

# %% [markdown] {"id":"a9HLJCe11Ejp"}
# ### Region

# %% [code] {"id":"uYDr1qPs1Ejp","outputId":"8454289d-9402-4b10-fcd5-02a4a92f5257","jupyter":{"outputs_hidden":false}}
len(data.region.unique())

# %% [code] {"id":"KpzORxYR1Ejp","outputId":"ae8b3305-a827-4c39-e3bf-7f6e21aa5f44","jupyter":{"outputs_hidden":false}}
data.groupby('region').size()

# %% [markdown] {"id":"1c1CNzVM1Ejp"}
# **There are ~338 observations from each region, dataset seems balanced, and there are 54 regions.**

# %% [markdown] {"id":"QxWV0XeB1Ejp"}
# ### The average prices by regions

# %% [code] {"id":"juGYjgJx1Ejp","outputId":"d5a8885b-47bd-4e33-9746-2b5b663739b8","jupyter":{"outputs_hidden":false}}
plt.figure(figsize=(12,5))
plt.title("Distribution Price")
ax = sns.distplot(data["AveragePrice"], color = 'g')

# %% [code] {"id":"tuWlghCR1Ejp","outputId":"c9a8ce54-dc57-44ab-894a-a1da558f8a36","jupyter":{"outputs_hidden":false}}
sns.boxplot(y="type", x="AveragePrice", data=data, palette = 'pink');

# %% [code] {"id":"1eaifnao1Ejq","outputId":"44df15b3-4e44-43e6-e3c6-f9d99baeb211","jupyter":{"outputs_hidden":false}}
#Weight distribution of prices
fig, ax = plt.subplots()
fig.set_size_inches(10,5)
sns.violinplot(data.dropna(subset = ['AveragePrice']).AveragePrice);

# %% [markdown] {"id":"jt_wjfEZ1Ejq"}
# Organic avocados are more expensive. This is obvious, because their cultivation is more expensive and we all love natural products and are willing to pay a higher price for them. But it is likely that the price of avocado depends not only on the type. Let's look at the price of avocado from different regions in different years. Let's start with organic avocados.

# %% [code] {"id":"SPzc75LW1Ejq","outputId":"bb57daa6-9e7e-4714-c5c8-649b8396d042","jupyter":{"outputs_hidden":false}}
plt.figure(figsize=(15,15))

plt.title("Avgerage Price of Avocado by Region")

sns.barplot(x="AveragePrice",y="region",data=data)

plt.show()

# %% [markdown] {"id":"WJa3Roye1Ejq"}
# Seems there are some regions which are US States (say California) and US Cities (say San Francisco) of that State or just Cities. Also there is a region as "TotalUS"; "West".

# %% [code] {"id":"fyexiquD1Ejq","outputId":"428e88d6-d2fb-4458-a990-6b25d68f6b6e","scrolled":true,"jupyter":{"outputs_hidden":false}}
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

# %% [markdown]
# **The most expensive avocados can be found in ['HartfordSpringfield', 'SanFrancisco', 'NewYork', 'Philadelphia', 'Sacramento']** 
# 
# **The cheapest avocados can be found in ['Houston', 'DallasFtWorth', 'SouthCentral', 'CincinnatiDayton', 'Nashville']**

# %% [code] {"id":"wN0xYokU1Ejq","outputId":"dde65414-f795-4468-947b-9aeea156f72a","jupyter":{"outputs_hidden":false}}
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

# %% [markdown] {"id":"Pql1fmC41Ejr"}
# ### Dates & Seasonality check

# %% [markdown] {"id":"gwhMcHLE1Ejr"}
# We have two columns which are 'Date' and 'year', being year the extracted year of date. To make the analysis easier, let's extract day and month out of 'Date' and see each value separately. That way, we are also going to have two more potentially usefull columns: day and month

# %% [code] {"id":"1j6P1UDC1Ejr","outputId":"20b7f9f1-21cd-4712-ad52-24661ae09f69","jupyter":{"outputs_hidden":false}}
from datetime import datetime
data['Date'] = data['Date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))

data['month'] = data['Date'].dt.month
data['day'] = data['Date'].dt.day
# monday = 0
data['day of week'] = data['Date'].dt.dayofweek
dates = ['year', 'month', 'day', 'day of week']
data[dates]

# %% [code] {"id":"2hgRUiue1Ejr","outputId":"1b1977f7-acac-44c5-a6e4-2aa915e275d9","jupyter":{"outputs_hidden":false}}
fig, ax = plt.subplots(2,2, figsize=(20,10))

sns.countplot('year', data=data, ax=ax[0,0], palette='BuGn_r')
sns.countplot('month', data=data, ax=ax[0,1], palette='BuGn_r')
sns.countplot('day', data=data, ax=ax[1,0], palette='BuGn_r')
sns.countplot('day of week', data=data, ax=ax[1,1], palette='BuGn')

plt.show()

# %% [markdown] {"id":"PSpCV6cx1Ejr"}
# **Year**
# 
# - 2015, 2016, 2017 have almost the same values
# - 2018 is the lowest, looks like the avocados should have ended in the begining of 2018
# 
# **Month**
# 
# - Shows a descending pattern, This could be because of the same reason as year: 2018 ended in the begging of the year and, therefore, the first months have more entries
# 
# **Day & day of week**
# 
# - We can see that the day chart has a repeating trend, and this is because of the day that the data was always recorded: day 6 (Sunday).
# - The data was, therefore, recorded weekly, 'day of week' becomes redundant and we can eliminate it.

# %% [code] {"id":"jD7Zd5Mm1Ejr","jupyter":{"outputs_hidden":false}}
data.drop('day of week', axis=1, inplace=True)

# %% [markdown] {"id":"VcqoGhop1Ejs"}
# <font color ='blue'>
# 
# - type' has to categories and is balanced, could be used as a classifier in model building
# - 'region' has 54 unique values and is perfectly balanced, could be hot encoded for model building
# - 'avg' price shows and pretty normal distribution and looks tentative for target variable for regression model
# - units sold columns show similar data which is similarly distributed, log formulas could be used to increase model performance
# - 'dates' is evenly distributed till 2018 and shows that the data was recorded on a weekly basis every Sunday

# %% [code] {"id":"r1JaaP-x1Ejs","outputId":"66635663-518f-4f3f-eb61-f691c0c69ce5","jupyter":{"outputs_hidden":false}}
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

# %% [markdown] {"id":"DfWEKW5A1Ejs"}
# <font color ='blue'>
# 
# - From the graphic we can tell that the, first of all, average price and total volume move in different direction
# - Total volume has a spike at the beginning of the year. On the other hand, average price drops at the beginning of the year
# - These drops and spikes are a sign of seasonality and that could help in forecasting

# %% [code] {"id":"dcJcrfVn1Ejs","outputId":"8f77af6d-4e6b-493f-ea95-f6cd897a70b0","jupyter":{"outputs_hidden":false}}
fig, ax = plt.subplots(1, 2, figsize=(12,5))

data['quarter'] = data['Date'].dt.quarter


sns.barplot(x='quarter', y='Total Volume', data=data, palette='Greens_r', ci=None, ax=ax[0])
sns.barplot(x='quarter', y='AveragePrice', data=data, palette='Greens_r', ci=None, ax=ax[1])


plt.show()

quarter = data.groupby('quarter')[['Total Volume', 'AveragePrice']].mean()
display(quarter)

# %% [markdown] {"id":"0BtNIQQI1Ejs"}
# - So we see that in the first quarter of the year sales are better than in other quarters and prices are the lowest.
# - After the first quarter, sales decrease and prices grow. Given the popularity of avos, businesses should be considering importing more avos when they are not produced in the country, a big oportunity for business-men from both countries.

# %% [code] {"scrolled":true,"jupyter":{"outputs_hidden":false}}
data.head()

# %% [markdown] {"id":"1WFXIJ8y1Ejs"}
# ### Type

# %% [code] {"id":"IAXdclzm1Ejs","outputId":"8076b197-1c4b-4a06-ca1b-3c122b3af272","jupyter":{"outputs_hidden":false}}
print(len(data.type.unique()))

data.groupby('type').size()

# %% [markdown] {"id":"xVCEMx2v1Ejt"}
# Types of avocados are also balanced since the ratio is nearly 0.5 each.

# %% [markdown] {"id":"Rmurq3nB1Ejt"}
# ### The average prices of avocados by types

# %% [code] {"id":"KaLStskd1Ejt","outputId":"4ed9f154-1b0c-4366-b3b4-28698860fde6","jupyter":{"outputs_hidden":false}}
plt.figure(figsize=(5,7))

plt.title("Avg.Price of Avocados by Type")

sns.barplot(x="type",y="AveragePrice",data= data)

plt.show()

# %% [code] {"id":"vAaWos4i1Ejt","outputId":"d9de83d0-2c50-44f4-ee38-c29f94f0b3c1","jupyter":{"outputs_hidden":false}}
plt.figure(figsize=(18,10))
sns.lineplot(x="month", y="AveragePrice", hue='type', data=data);

# %% [code] {"id":"58HbYngj1Ejt","outputId":"d6f9b856-4493-4dba-cedd-1941a8560139","jupyter":{"outputs_hidden":false}}
data['month'].head()

# %% [code] {"id":"vi36M7Ya1Ejt","outputId":"7af47954-6c20-4dca-fb7d-d9e8a228fe63","jupyter":{"outputs_hidden":false}}
data['month'] = data['month'].replace({1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 
                                   6: 'June', 7: 'July', 8: 'August', 9: 'September', 10: 'October', 
                                   11: 'November', 12: 'December'})
ax = sns.catplot(x="month", y="AveragePrice", hue="type", 
            kind="box", data=data, height=8.5, linewidth=2.5, aspect=2.8,palette="Set2");

# %% [code] {"id":"FFxgUL5I1Ejt","outputId":"f4b97311-7a06-45d1-9344-b3adffde77cb","jupyter":{"outputs_hidden":false}}
plt.figure(figsize=(18,10))
sns.lineplot(x="month", y="AveragePrice", hue='year',  data=data)
plt.show()

# %% [code] {"id":"L4LAMJ_p1Ejt","outputId":"7f4980e7-fb03-424d-8453-7aebfdb82bc6","jupyter":{"outputs_hidden":false}}
fig, ax = plt.subplots(1, 2, figsize=(14,5))

sns.barplot(x='type', y='AveragePrice', data=data, palette='Set3', ax=ax[0])
sns.barplot(x='type', y='Total Volume', data=data, palette='Set3', ax=ax[1], estimator=sum, ci=None)
plt.show()

display(data.groupby('type')['AveragePrice'].mean())
display(data.groupby('type')['Total Volume'].sum())

# %% [markdown] {"id":"Q1_8pzdu1Eju"}
# **Convential is cheaper than organic, but surprisingly, conventional destroyed organic sells. So conventional avos are performing quite well and organic are being left behind, but is organic at least geaining popularity?**

# %% [code] {"id":"B9-15Ene1Eju","outputId":"915fa84a-bc63-431a-e48b-ccfe7aa35adb","jupyter":{"outputs_hidden":false}}
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

# %% [markdown] {"id":"gbmUauod1Eju"}
# Is hard to tell from the graphic alone but if we sum every percentage change we find that organic has a bigger growth overall with 200.48 against 137.02 of conventional.
# 
# Let's add some business strategy concepts to refine strategy and conclusions here.
# 
# The BCG matix is a model that evaluates how a business is performing according its growth and market share. It has for dimensions:
# 
# Dogs: These are products with low growth or market share.
# Question marks or Problem Child: Products in high growth markets with low market share.
# Stars: Products in high growth markets with high market share.
# Cash cows: Products in low growth markets with high market share.
# Organic might be having way smaller sales than conventional, but its growing rate (higher than conventional) is a good sign to keep producing the organic avos and it already has a market. This is a healthy indicator for businesses.Then, organic is a Star in the BCG matrix. A suggestion would then be to have a business growth strategy with them: technologies and methods that produce more and cheaper, promotion and importations.
# 
# Conventional avos are too succesfull and have an already stablished business infrastructure. Therefore, conventional are Cash cows in the BCG matrix, and businesses should keep producing them at the same or higher rate.

# %% [code] {"id":"DGkRftDn1Eju","outputId":"dd1b374a-0086-45f2-ae81-ac3ec20c0f63","jupyter":{"outputs_hidden":false}}
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

# %% [code] {"id":"h58_BWwM1Eju","outputId":"e2c72ff7-8371-43f2-cbb6-cb4f560cac0e","jupyter":{"outputs_hidden":false}}
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

# %% [code] {"id":"90E9o-yG1Eju","outputId":"aa94f678-1a17-4b5f-84a9-0f7b1e92e779","jupyter":{"outputs_hidden":false}}
g = sns.factorplot('AveragePrice','region',data=data,
                   hue='year',
                   size=18,
                   aspect=0.7,
                   palette='magma',
                   join=False,
              )

# %% [markdown] {"id":"QmnpSxzh1Eju"}
# Oh San Francisco, 2017..... In 2017, organic avocados were very expensive :( Search in Google gave result on this question. In 2017, there was a shortage of avocados. That explains the price increase!

# %% [code] {"id":"C5ho1VNV1Ejv","outputId":"f9d2650f-e13e-4283-9de5-d07c0ba1a361","jupyter":{"outputs_hidden":false}}
mask = data['type']=='conventional'
g = sns.factorplot('AveragePrice','region',data=data[mask],
                   hue='year',
                   size=13,
                   aspect=0.8,
                   palette='magma',
                   join=False,
              )

# %% [markdown] {"id":"ZyJyojfj1Ejv"}
# For obvious reasons, prices are lower. The situation with the price increase in 2017 also affected this type of avocado.
# 
# Organic avocado type is more expensive. And avocado is generally more expensive with each passing year

# %% [code] {"id":"ddN02GTq1Ejv","outputId":"cc5aaef3-c9f7-44f0-c9c4-a48c2d5e7d21","jupyter":{"outputs_hidden":false}}
numbers = list(data.select_dtypes(['float64', 'int64']).keys())

# removing years
numbers.remove('year')

data[numbers].hist(figsize=(20,10), color='green', edgecolor='white')

plt.show()

display(data[numbers].describe())

# %% [markdown] {"id":"SS8NBnDI1Ejv"}
# <font color='blue'> **Average Price**
# 
# - Is the most normal distribution. Mean and median are really closed, which means the distribution is not severly influenced by outliers. Still, it is a bit skewed to the right, the mean being bigger than the median reflects that. Remaining features
# - The remaining features are severely influenced by outliers, most of the values are located in the first bin of the histograms and the meean is way bigger than the median.
# - These features seem to follow the same distribution, which makes sense since the information (quantity sold) is similar
# 
# Lets take the outliers out of the quantities to see if we can find a more normal distribution <font>

# %% [code] {"id":"vIPqO9C31Ejv","outputId":"2579c669-5e9d-4a6c-cdaa-438607f5c00e","jupyter":{"outputs_hidden":false}}
data_o = data[data['Total Volume']<50000]
data_o[numbers].hist(figsize=(20,10), color='green', edgecolor='white')

plt.show()

# %% [markdown] {"id":"Se0FVXld1Ejv"}
# These kind of distributions, where most of the values are located in lower values and then descends, is really common and could be represented in a different way through log formulas to make it more 'normal' and useful for a model, like regression models, without getting rid of outliers.
# 
# A example below with Total Volume.
# 
# Refer **Outlier Detection sction** for details

# %% [code] {"id":"8BiihFoA1Ejv","outputId":"cea6aaed-e3a7-4b06-c4ad-78e62d941764","jupyter":{"outputs_hidden":false}}
TotalLog = np.log(data['Total Volume'] + 1)
TotalLog.hist(color='green', edgecolor='white');

# %% [markdown] {"id":"QP_Azn8C1Ejw"}
# ### Correlation

# %% [code] {"id":"hEXzdYIx1Ejw","outputId":"4531d3eb-dd47-4e6b-d3bc-504084019b26","jupyter":{"outputs_hidden":false}}
plt.figure(figsize=(12,6))
sns.heatmap(data.corr(),cmap='coolwarm',annot=True);

# %% [markdown] {"id":"f-6iBI-Q1Ejw"}
# **Observation :**
# 
# There is a high correlation between pairs:
# 
# 4046 & total volume (0.98)
# 4225 & total volume (0.97)
# 4770 & total volume (0.87)
# total bags & total volume (0.96)
# small bags & total bags (0.99)
# etc
# 
# 4046 avocados are the most preferred/sold type in the US and customers tend to buy those avocados as bulk, not bag.
# 
# Retailers want to increase the sales of bagged avocados instead of bulks. They think this is more advantageous for them.
# Total Bags variable has a very high correlation with Total Volume (Total Sales) and Small Bags, so we can say that most of the bagged sales comes from the small bags.

# %% [code] {"id":"VUBdtcX-1Ejw","jupyter":{"outputs_hidden":false}}
from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
dicts = {}

label.fit(data.type.drop_duplicates()) 
dicts['type'] = list(label.classes_)
data.type = label.transform(data.type)

# %% [code] {"id":"vjYj-I7u1Ejw","outputId":"caa28a52-13c6-4eee-a87a-fa0194c0668f","jupyter":{"outputs_hidden":false}}
cols = ['AveragePrice','type','year','Total Volume','Total Bags']
cm = np.corrcoef(data[cols].values.T)
sns.set(font_scale = 1.7)
hm = sns.heatmap(cm,cbar = True, annot = True,square = True, fmt = '.2f', annot_kws = {'size':15}, yticklabels = cols, xticklabels = cols)

# %% [markdown] {"id":"77OWcT6g1Ejw"}
# The price of avocado is influenced by the type. Logically. We also see a strong correlation between the features: "Total Bags" and "Total Volume". Also, if you look at the correlation of all the features, you will notice that strongly correlated Small Bags,Large Bag. It is logical but can create problems if we go to predict the price of avocado.

# %% [code] {"id":"_krLcWOe1Ejw","outputId":"2673d2bf-9b63-4ea5-9978-1bd49a92db63","jupyter":{"outputs_hidden":false}}
fig, ax = plt.subplots(1, 2,figsize=(40,15))

data_o = data[data['Total Volume']<50000]

sns.heatmap(data.corr(), vmin=-1, vmax=1, cmap=sns.diverging_palette(20, 220, as_cmap=True), annot=True, ax=ax[0])
ax[0].set_title('With outliers', fontsize=25)

sns.heatmap(data_o.corr(), vmin=-1, vmax=1, cmap=sns.diverging_palette(20, 220, as_cmap=True), annot=True, ax=ax[1])
ax[1].set_title('Without outliers', fontsize=25)

plt.show()

# %% [markdown] {"id":"gy6keRQD1Ejx"}
# - We are going to take the strongest relationship out of the volume variable and the strongest out of a date variable
# - We are going to take the relationships with AveragePrice, out of both heatmaps, since is our target variable for the regression model

# %% [code] {"id":"99QqPhFP1Ejx","outputId":"c02cccb7-50f7-40c2-dafc-5c25a81da95b","jupyter":{"outputs_hidden":false}}
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

# %% [markdown] {"id":"-ZGOYVN71Ejx"}
# <font color='blue'>
#     
# - An important insight here is that we can't take the outliers out since all of them correspond to the conventional type, which means that conventional avocados sell way more than organic avocados
#     
# - There doesn't seem to be a relationship between month and AveragePrice, what we can see in this graph is that the average price of conventional avocados is way smaller that the organic. We are going to take a closer look at this in the further sections
#     
# - There is an expected decreasing trend for both types: the more units were sold, the less the average price is, we are going to take a closer look at this later as well.
#     
# - Perhaps a better way of representing the data is not by taking out the outliers but by normilizing the data, let's try that now with AveragePrice and 4046

# %% [code] {"id":"VxMRdbpH1Ejx","outputId":"c838fc2f-b1f7-4b9a-edd6-fdd8cf39e7f1","jupyter":{"outputs_hidden":false}}
scaler = Normalizer()
scaler.fit(data[['4046', 'AveragePrice']].values)
data['4046_scaled'] = scaler.transform(data[['4046', 'AveragePrice']].values)[:,0]
data['AveragePrice_scaled'] = scaler.transform(data[['4046', 'AveragePrice']].values)[:,1]

sns.regplot(x='4046_scaled', y='AveragePrice_scaled', data=data, color='g')
plt.show()

# %% [markdown] {"id":"26YCSulR1Ejx"}
# **We now know that both the regression and classification is possible since there is a clear tendency**

# %% [markdown] {"id":"_hbvkfM51Ejx"}
# <font color='blue'> *As we already see the field descriptions, so for our training we are interested only in fields as below*

# %% [code] {"id":"Z310bKCf1Ejx","jupyter":{"outputs_hidden":false}}
# Specifying dependent and independent variables

X = data[['4046', '4225', '4770', 'Small Bags', 'Large Bags', 'XLarge Bags', 'type', 'year', 'region']]
Y = data['AveragePrice']
y=np.log1p(Y)

# %% [code] {"id":"FyaD0MAG1Ejx","outputId":"d144a8fd-fe9a-4e4b-dbcf-20fe083be3c8","jupyter":{"outputs_hidden":false}}
X.head()

# %% [code] {"id":"PUNY7TDP1Ejy","outputId":"ca2db81c-a6af-4b8e-f907-07e46f39879f","jupyter":{"outputs_hidden":false}}
Y.head()

# %% [markdown] {"id":"BjKuNEe31Ejy"}
# ### Labeling the categorical variables

# %% [code] {"id":"NVlk5fdx1Ejy","outputId":"f7886bef-5f4f-4949-e1f1-e0e0fa5182aa","jupyter":{"outputs_hidden":false}}
# X_labelled = pd.get_dummies(X[["type","region"]], drop_first = True)
# X_labelled.head()

X = pd.get_dummies(X, prefix=["type","region"], columns=["type","region"], drop_first = True)
X.head()

# %% [code] {"id":"I9ytEXEa1Ejy","outputId":"d2213821-54be-4745-c80b-37a43bd16c8c","jupyter":{"outputs_hidden":false}}
print(X.columns)

# %% [markdown] {"id":"H_Oryljd1Ejy"}
# ## Split into Train and Valid set

# %% [code] {"id":"aSXcfc151Ejy","jupyter":{"outputs_hidden":false}}
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import mean_squared_error

# %% [code] {"id":"OHPhyDT31Ejy","jupyter":{"outputs_hidden":false}}
X_train, X_valid, y_train, y_valid = train_test_split(X, Y, test_size = 0.3, random_state = 99)

# %% [code] {"id":"aL1jnfP61Ejz","outputId":"144736dc-37f5-49b8-a140-8a9afd055e51","jupyter":{"outputs_hidden":false}}
X_train.shape, X_valid.shape, y_train.shape, y_valid.shape

# %% [markdown] {"id":"PkYBZg5t1Ejz"}
# ## Training the Model

# %% [markdown] {"id":"SDI9F6IV1Ejz"}
# ### Multiple Linear Regression

# %% [code] {"id":"ulvGLWNc1Ejz","outputId":"8fd71631-1b9e-4737-ca28-08712b282609","jupyter":{"outputs_hidden":false}}
lr = LinearRegression()
lr.fit(X_train,y_train)

print("R2 of Linear Regresson:", lr.score(X_train,y_train) )
print("----- Prediction Accuracy-----")
print('MAE: ',metrics.mean_absolute_error(y_valid, lr.predict(X_valid)))
print('MSE: ',metrics.mean_squared_error(y_valid, lr.predict(X_valid)))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_valid, lr.predict(X_valid))))

# %% [markdown]
# **R2 of Linear Regresson: 0.56176162598828**
# 
# **----- Prediction Accuracy-----**
# 
# **MAE:  0.20301652029791611**
# 
# **MSE:  0.07278245038216843**
# 
# **RMSE: 0.2697822276988765**

# %% [code] {"id":"meOU2RmJ1Ejz","outputId":"edfa736f-d72b-49d7-c22a-0622aaff7e57","jupyter":{"outputs_hidden":false}}
# Creating a Histogram of Residuals
plt.figure(figsize=(6,4))
sns.distplot(y_valid - lr.predict(X_valid))
plt.title('Distribution of residuals')
plt.show()

# %% [code] {"id":"_l0Z6dte1Ejz","outputId":"1d5c5ea2-8502-451d-efea-d02a29d3b3aa","jupyter":{"outputs_hidden":false}}
plt.scatter(y_valid,lr.predict(X_valid));

# %% [markdown] {"id":"UOLEYfmc1Ejz"}
# ## Support Vector Regression

# %% [code] {"id":"fyGZP9Ze1Ejz","jupyter":{"outputs_hidden":false}}
from sklearn.svm import SVR

# %% [markdown] {"id":"4gtTitfG1Ej0"}
# #### *let's first choose the best kernel for our data out of provided kernels.*

# %% [code] {"id":"PPz1xOws1Ej0","jupyter":{"outputs_hidden":false}}
#clf = svm.SVR(kernel = 'linear')
#clf.fit(X_train, y_train)
#confidence = clf.score(X_train, y_train)
#print(k,confidence)

# %% [code] {"id":"XvhgbF5c1Ej0","jupyter":{"outputs_hidden":false}}
#for k in ['linear','poly','rbf','sigmoid']:
#    print("Running for k as ", k)
#    clf = svm.SVR(kernel=k)
#    clf.fit(X_train, y_train)
#    confidence = clf.score(X_train, y_train)
#    print(k,confidence)

# %% [markdown] {"id":"n8riKyyk1Ej0"}
# ### Parameter Tuning or Hyperparameter

# %% [markdown] {"id":"SaVRQvSn1Ej0"}
# Intuitively, the *gamma* defines how far the influence of a single training example reaches, with low values meaning ‘far’ and high values meaning ‘close’.
# 
# The *C* parameter trades off correct classification of training examples against maximization of the decision function’s margin.
# 
# For larger values of *C*, a smaller margin will be accepted if the decision function is better at classifying all training points correctly.
# 
# A lower *C* will encourage a larger margin, therefore a simpler decision function, at the cost of training accuracy.
# 
# In other words *C* behaves as a **regularization parameter** in the SVM.

# %% [code] {"id":"MpvdIXBX1Ej0","outputId":"b7b80135-9b16-4b4a-f2a4-8c8c3587b197","jupyter":{"outputs_hidden":false}}
svr = SVR(kernel='rbf', C=1, gamma= 0.5)   # Parameter Tuning to get the best accuracy

svr.fit(X_train,y_train)
print(svr.score(X_train,y_train))

# %% [code] {"id":"CxwNjybH1Ej1","jupyter":{"outputs_hidden":false}}
from math import sqrt

# %% [code] {"id":"9lzJ_glQ1Ej1","outputId":"a78f0f03-921f-4865-f747-0cbde79c3644","jupyter":{"outputs_hidden":false}}
# calculate RMSE
error = sqrt(metrics.mean_squared_error(y_valid,svr.predict(X_valid))) 
print('RMSE value of the SVR Model is:', error)

# %% [code] {"id":"-Nh-j9Dd1Ej1","outputId":"4547a5ca-87cd-43b2-a50b-1aa7c4c12721","jupyter":{"outputs_hidden":false}}
# Creating a Histogram of Residuals
plt.figure(figsize=(6,4))
sns.distplot(y_valid - svr.predict(X_valid))
plt.title('Distribution of residuals')
plt.show()

# %% [code] {"id":"F16nyUaY1Ej1","outputId":"1a7eb878-1ee2-4434-b280-64216ba40cc7","jupyter":{"outputs_hidden":false}}
plt.scatter(y_valid,svr.predict(X_valid));

# %% [code] {"id":"NY29_2qs1Ej1","outputId":"6dc45e03-8f30-454d-e77b-3d0e1be57b2f","jupyter":{"outputs_hidden":false}}
# Linear Regression RMSE : 
print('RMSE value of the Linear Regr : ',round(np.sqrt(metrics.mean_squared_error(y_valid, lr.predict(X_valid))),4))

# SVR RMSE               : 
print('RMSE value of the SVR Model   : ',round(np.sqrt(metrics.mean_squared_error(y_valid, svr.predict(X_valid))),4))

# %% [code] {"jupyter":{"outputs_hidden":false}}
data=data.drop(['Date'], axis=1)

# %% [code] {"jupyter":{"outputs_hidden":false}}
data_dt=data # for decision tree alogorithm

# %% [code] {"id":"CyOOXR-u1Ej1","jupyter":{"outputs_hidden":false}}
data=data.drop(['year_month'], axis=1)

# %% [markdown] {"id":"ZxGIE9k_1Ej2"}
# ## Train & Validation

# %% [markdown] {"id":"TkhVyWdP1Ej2"}
# As we are predicting the price of the avocados we are going to put the prices column in the Y and rest of the data in X

# %% [code] {"id":"XSWKdjdX1Ej2","jupyter":{"outputs_hidden":false}}
X=datam.drop('AveragePrice',1)
y=datam['AveragePrice']

# %% [code] {"id":"TkXZwwkG1Ej2","outputId":"aa8173f0-7512-407e-8882-5d8db0a7d51e","jupyter":{"outputs_hidden":false}}
print('shape of X and y respectively :',X.shape,y.shape)

# %% [code] {"jupyter":{"outputs_hidden":false}}
X.head()

# %% [markdown] {"id":"R2-fURz21Ej2"}
# #### *performing a 80-20 train test split over the dataset.*

# %% [code] {"id":"3LOrMRo61Ej2","jupyter":{"outputs_hidden":false}}
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

# %% [code] {"id":"VJv0cK3b1Ej2","outputId":"5fbb7289-8704-4108-ad73-2c565578d176","jupyter":{"outputs_hidden":false}}
print('shape of X and y respectively(train) :',X_train.shape,y_train.shape)
print('shape of X and y respectively(test) :',X_test.shape,y_test.shape)

# %% [code] {"jupyter":{"outputs_hidden":false}}
cols=X_train.columns

# %% [markdown] {"id":"Yw5KWcJJ1Ej3"}
# ### Preprocessing

# %% [markdown] {"id":"wf2zbjw41Ej3"}
# ### <font color="green">Encoding

# %% [markdown] {"id":"qfTwZDeu1Ej3"}
# #### *Encoding all the categorical columns to dig deep into the data.*

# %% [code] {"id":"WWzvrPPr1Ej3","jupyter":{"outputs_hidden":false}}
scaler=LabelEncoder()

# %% [code] {"id":"hpeVJ-qc1Ej3","outputId":"39303745-a486-4610-a24e-70bc373d49af","jupyter":{"outputs_hidden":false}}
for col in X_train.columns:
    if datam[col].dtype=='object':
        X_train[col]=scaler.fit_transform(X_train[col])
        X_test[col]=scaler.transform(X_test[col])

# %% [code] {"id":"t4KKpYX11Ej3","outputId":"a100288f-29f8-483e-8c20-99f5d15b489a","jupyter":{"outputs_hidden":false}}
X_train.head()

# %% [code] {"jupyter":{"outputs_hidden":false}}
X_train.shape

# %% [markdown] {"id":"Mz3-mHdS1Ej4"}
# ### <font color="green">Variance thresholding

# %% [markdown] {"id":"mT_SoY-x1Ej4"}
# #### *Now after encoding the dataframe we have to omit the columns which are not contributing any pattern or key for finding good accuracy. That means we are going to drop the columns which have less variance than 0.1*

# %% [code] {"id":"dXzttupa1Ej4","jupyter":{"outputs_hidden":false}}
scaler=VarianceThreshold(0.1)

# %% [code] {"id":"fF0S6Mjv1Ej4","outputId":"a60fc8b9-f200-43b0-8f48-0d8434133990","scrolled":true,"jupyter":{"outputs_hidden":false}}
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

# %% [code] {"id":"GLRdWNk41Ej4","jupyter":{"outputs_hidden":false}}
print('shape of X (train) :',X_train.shape)
print('shape of X (test) :',X_test.shape)

# %% [markdown] {"id":"fFWozmaJ1Ej4"}
# ### <font color="green">Scaling

# %% [code] {"id":"aXRLGb1U1Ej4","jupyter":{"outputs_hidden":false}}
plt.plot(X_train[0]);

# %% [markdown] {"id":"KpZqKwrG1Ej4"}
# This graph shows that the every single feature has different value ranges. So we need to scale the data for better performances.

# %% [code] {"id":"9DbMBeIO1Ej5","jupyter":{"outputs_hidden":false}}
scaler=StandardScaler()

# %% [code] {"id":"RPAL02za1Ej5","jupyter":{"outputs_hidden":false}}
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

# %% [code] {"id":"hcbB5vV51Ej5","jupyter":{"outputs_hidden":false}}
print("Type of X_train :",type(X_train))
print("Type of X_test :",type(X_test))

# %% [markdown] {"id":"cTNM1W9I1Ej5"}
# As a result of using the **variance thershold** and **standard scaler** of *sklearn module* the pandas dataframe changed in numpy ndarray. So we are going to convert them into pandas dataframe.

# %% [code] {"id":"FUK0hkAQ1Ej5","jupyter":{"outputs_hidden":false}}
X_train=pd.DataFrame(X_train,columns=cols)
X_train.head()

# %% [code] {"id":"js2HBPML1Ej5","jupyter":{"outputs_hidden":false}}
X_test=pd.DataFrame(X_test,columns=cols)
X_test.head()

# %% [code] {"id":"cXyUpR6R1Ej5","jupyter":{"outputs_hidden":false}}
print('Type of X_train and X_test :',type(X_train),type(X_test))

# %% [markdown] {"id":"XuBGNoLS1Ej5"}
# ## Pipeline

# %% [markdown] {"id":"ELdRv2nj1Ej6"}
# As this is a regression problem we are going to use famous regression models -
# 
#     Polynomial Regression
#     RandomForest Regression
#     Ridge Regression
#     Lasso Regression
#     Bayesian Ridge Regression

# %% [code] {"id":"QGAS-apP1Ej6","jupyter":{"outputs_hidden":false}}
actr=[]
acts=[]
lstr=[]
lsts=[]

# %% [markdown] {"id":"svdg1vLJ1Ej6"}
# ### Polynomial Regression

# %% [code] {"id":"LOCHL8WF1Ej6","jupyter":{"outputs_hidden":false}}
loss=[]
val_loss=[]

# %% [code] {"id":"01C5JejH1Ej6","jupyter":{"outputs_hidden":false}}
for i in range(2,5):
  clf=make_pipeline(PolynomialFeatures(i),LinearRegression())
  clf.fit(X_train,y_train)
  y_pr=clf.predict(X_test)
  y_x=clf.predict(X_train)
  loss.append(mean_squared_error(y_train,y_x))
  val_loss.append(mean_squared_error(y_test,y_pr))

# %% [code] {"id":"NQr4Q6CU1Ej6","jupyter":{"outputs_hidden":false}}
plt.title('Model Loss')
plt.xlabel('degree')
plt.ylabel('MSE loss')
plt.plot(range(2,5),loss/np.mean(loss),label='train loss')
plt.plot(range(2,5),val_loss/np.mean(val_loss),label='validation loss')
plt.legend()
plt.show()

# %% [markdown] {"id":"S7zM7ruy1Ej6"}
# We can say that degree =3 is appropriate for the model as the validation has too little loss and after that it reaches to a very high peak. That means after degree>3 ,the model is overfitting.

# %% [code] {"id":"6nrknBUs1Ej6","jupyter":{"outputs_hidden":false}}
print('Train loss and validation loss of the polynomial function model :',loss[1],'and',val_loss[1])

# %% [code] {"id":"CRLO61dy1Ej7","jupyter":{"outputs_hidden":false}}
clf=make_pipeline(PolynomialFeatures(3),LinearRegression())
clf.fit(X_train,y_train)
print('train accuracy :',clf.score(X_train,y_train))
print('test accuracy :',clf.score(X_test,y_test))

# %% [markdown] {"id":"9qW-T1xW1Ej7"}
# Though degree=3 has the best accuracy over the polynomial function still it is very low and we can see that the validation accurcay is less than 50%. So, it is not a good model.

# %% [code] {"id":"5Erh3LmV1Ej7","jupyter":{"outputs_hidden":false}}
actr.append(clf.score(X_train,y_train))
acts.append(clf.score(X_test,y_test))
lstr.append(loss[1])
lsts.append(val_loss[1])

# %% [markdown] {"id":"rTHI0YfC1Ej7"}
# ### Random Forest Regression

# %% [code] {"id":"4uUUkOpl1Ej7","jupyter":{"outputs_hidden":false}}
clf=RandomForestRegressor(random_state=0)

# %% [code] {"id":"Dyd-460W1Ej7","jupyter":{"outputs_hidden":false}}
clf.fit(X_train,y_train)
y_tr1=clf.predict(X_train)
y_pr=clf.predict(X_test)

# %% [code] {"id":"54senaHR1Ej7","jupyter":{"outputs_hidden":false}}
print('train data accuracy :',clf.score(X_train,y_train))
print('test data accuracy :',clf.score(X_test,y_test))
print('loss of train data :',mean_squared_error(y_train,y_tr1))
print('loss of test data :',mean_squared_error(y_test,y_pr))

# %% [markdown] {"id":"Xbmdg3fq1Ej8"}
# So we can see the RFR really predicts the model very well and gives a quite accurate prediction.

# %% [code] {"id":"ac7e_jme1Ej8","jupyter":{"outputs_hidden":false}}
actr.append(clf.score(X_train,y_train))
acts.append(clf.score(X_test,y_test))
lstr.append(mean_squared_error(y_train,y_tr1))
lsts.append(mean_squared_error(y_test,y_pr))

# %% [markdown] {"id":"I9Vv3XhA1Ej8"}
# ### Ridge Regression

# %% [code] {"id":"EWmeKefd1Ej8","jupyter":{"outputs_hidden":false}}
loss=[]
val_loss=[]

# %% [code] {"id":"StkzqHYi1Ej8","jupyter":{"outputs_hidden":false}}
for i in range(1,11):
  clf=Ridge(random_state=0,alpha=i/100.0)
  clf.fit(X_train,y_train)
  y_pr=clf.predict(X_test)
  y_x=clf.predict(X_train)
  loss.append(mean_squared_error(y_train,y_x))
  val_loss.append(mean_squared_error(y_test,y_pr))

# %% [code] {"id":"EKkHGxqb1Ej8","jupyter":{"outputs_hidden":false}}
plt.title('Model Loss')
plt.xlabel('alpha')
plt.ylabel('MSE loss')
plt.plot(np.arange(1,11,1)/100,loss/np.mean(loss),label='train loss')
plt.plot(np.arange(1,11,1)/100,val_loss/np.mean(val_loss),label='validation loss')
plt.legend()
plt.show()

# %% [markdown] {"id":"_ptdsNMF1Ej9"}
# So we can say that the increase in alpha also affecting the model badly and giving us more loss than before.

# %% [code] {"id":"t9eoDJwk1Ej9","jupyter":{"outputs_hidden":false}}
clf=Ridge(random_state=0,alpha=0.01)
clf.fit(X_train,y_train)
y_tr1=clf.predict(X_train)
y_pr=clf.predict(X_test)

print('train data accuracy :',clf.score(X_train,y_train))
print('test data accuracy :',clf.score(X_test,y_test))
print('loss of train data :',mean_squared_error(y_train,y_tr1))
print('loss of test data :',mean_squared_error(y_test,y_pr))

# %% [markdown] {"id":"jAd-bkP-1Ej9"}
# Still this model gives very bad stats in fitting.

# %% [code] {"id":"WUDsnPuG1Ej9","jupyter":{"outputs_hidden":false}}
actr.append(clf.score(X_train,y_train))
acts.append(clf.score(X_test,y_test))
lstr.append(mean_squared_error(y_train,y_tr1))
lsts.append(mean_squared_error(y_test,y_pr))

# %% [markdown] {"id":"HG1I2ReW1Ej9"}
# ### Lasso Regression

# %% [code] {"id":"A__HoLkk1Ej9","jupyter":{"outputs_hidden":false}}
loss=[]
val_loss=[]

# %% [code] {"id":"yFNHD6XR1Ej9","jupyter":{"outputs_hidden":false}}
for i in range(1,11):
  clf=Lasso(random_state=0,alpha=i/100.0)
  clf.fit(X_train,y_train)
  y_pr=clf.predict(X_test)
  y_x=clf.predict(X_train)
  loss.append(mean_squared_error(y_train,y_x))
  val_loss.append(mean_squared_error(y_test,y_pr))

# %% [code] {"id":"NabrSgza1Ej-","jupyter":{"outputs_hidden":false}}
plt.title('Model Loss')
plt.xlabel('alpha')
plt.ylabel('MSE loss')
plt.plot(np.arange(1,11,1)/100,loss/np.mean(loss),label='train loss')
plt.plot(np.arange(1,11,1)/100,val_loss/np.mean(val_loss),label='validation loss')
plt.legend()
plt.show()

# %% [markdown] {"id":"3SWweeiI1Ej_"}
# Lasso model also gives us similar results as we got in Ridge. We're taking the alpha=0.01

# %% [code] {"id":"-VTnSTy91Ej_","jupyter":{"outputs_hidden":false}}
clf=Lasso(random_state=0,alpha=0.01)
clf.fit(X_train,y_train)
y_tr1=clf.predict(X_train)
y_pr=clf.predict(X_test)

print('train data accuracy :',clf.score(X_train,y_train))
print('test data accuracy :',clf.score(X_test,y_test))
print('loss of train data :',mean_squared_error(y_train,y_tr1))
print('loss of test data :',mean_squared_error(y_test,y_pr))

# %% [code] {"id":"0U2nWKXw1Ej_","jupyter":{"outputs_hidden":false}}
actr.append(clf.score(X_train,y_train))
acts.append(clf.score(X_test,y_test))
lstr.append(mean_squared_error(y_train,y_tr1))
lsts.append(mean_squared_error(y_test,y_pr))

# %% [markdown] {"id":"8yiW_fVy1Ej_"}
# ## Bayesian Ridge Regression

# %% [code] {"id":"_C0T1qgy1Ej_","jupyter":{"outputs_hidden":false}}
clf=BayesianRidge()

# %% [code] {"id":"bsxh1a2x1EkA","jupyter":{"outputs_hidden":false}}
clf.fit(X_train,y_train)
y_tr1=clf.predict(X_train)
y_pr=clf.predict(X_test)

print('train data accuracy :',clf.score(X_train,y_train))
print('test data accuracy :',clf.score(X_test,y_test))
print('loss of train data :',mean_squared_error(y_train,y_tr1))
print('loss of test data :',mean_squared_error(y_test,y_pr))

# %% [code] {"id":"7uluzLld1EkA","jupyter":{"outputs_hidden":false}}
actr.append(clf.score(X_train,y_train))
acts.append(clf.score(X_test,y_test))
lstr.append(mean_squared_error(y_train,y_tr1))
lsts.append(mean_squared_error(y_test,y_pr))

# %% [markdown] {"id":"NP2KYlh21EkA"}
# ## Model Evaluation

# %% [code] {"id":"6MaJZd0m1EkA","jupyter":{"outputs_hidden":false}}
models=['Polynomial','Random Forest','Ridge','Lasso','Bayesian Ridge']

# %% [code] {"id":"l4NOd86T1EkA","jupyter":{"outputs_hidden":false}}
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

# %% [markdown] {"id":"RrV2DTPz1EkA"}
# **Model performance : Random Forest > Ridge > Bayesian Ridge > Lasso > Polynomial**
#                        
# After model evaluation we can conclude that, high dimensional data can not be fit well in low dimensional models and can give abrupt conclusions that may led to lower accuracy.
# 
# The RFR uses leafs that can reduce the dimensional complexity and generalize the model in a better approach. Thus it presents the best accuracy over the data.

# %% [markdown] {"id":"-d6a5tOX1EkA"}
# # 70:30 split

# %% [code] {"id":"4TcHLdG41EkA","jupyter":{"outputs_hidden":false}}
from sklearn.model_selection import train_test_split

trainflights, testflights, ytrain, ytest = train_test_split(data, y, train_size=0.7,test_size=0.3, random_state=0)

# %% [code] {"id":"-JrijFH-1EkB","jupyter":{"outputs_hidden":false}}
s = (trainflights.dtypes == 'object')
object_cols = list(s[s].index)

n = (trainflights.dtypes == ('float64','int64'))
numerical_cols = list(n[n].index)

# %% [code] {"id":"KhPpfG741EkB","outputId":"d6de980d-23d4-47e7-ef1f-1a2471185a8b","jupyter":{"outputs_hidden":false}}
#checking the columns containing categorical columns:
print(object_cols)

# %% [code] {"id":"8BKUR03C1EkB","jupyter":{"outputs_hidden":false}}
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

# %% [code] {"id":"unG9m3ke1EkB","outputId":"22f70afd-4c64-437b-b54d-7cc3895b8508","jupyter":{"outputs_hidden":false}}
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

# %% [code] {"id":"Drogdln11EkB","jupyter":{"outputs_hidden":false}}
#implementing the algo:
model = RandomForestRegressor(n_estimators=100, random_state=0, verbose=1)

#fitting the data to random forest regressor:
model.fit(trainflights, ytrain)

# %% [code] {"id":"keEE779w1EkB","jupyter":{"outputs_hidden":false}}
#predicting the test dataset:
preds = model.predict(testflights)
PercentError(preds, ytest)

# %% [code] {"id":"n6bheUSa1EkB","jupyter":{"outputs_hidden":false}}
#using linear regression:
LinearModel = LinearRegression()
LinearModel.fit(trainflights, ytrain)

# %% [code] {"id":"7v5CthiL1EkC","jupyter":{"outputs_hidden":false}}
#predicting on the test dataset:
LinearPredictions = LinearModel.predict(testflights)
PercentError(LinearPredictions, ytest)

# %% [markdown] {"id":"wmOfthNw1EkC"}
# # Decision-Tree

# %% [markdown] {"id":"X86SWxtj1EkC"}
# ## Prediction for Small bags & The average price

# %% [code] {"id":"S3LacoKR1EkC","jupyter":{"outputs_hidden":false}}
df=data[["year","Small Bags","Large Bags", "AveragePrice"]]

df = df.sample(n=50,replace=True)
#df=df.head(50)
df.tail()

# %% [code] {"id":"Y54mhavt1EkF","jupyter":{"outputs_hidden":false}}
y=df.iloc[:,1].values
x=df.iloc[:,-1].values

# %% [code] {"id":"m28ww4tm1EkG","jupyter":{"outputs_hidden":false}}
x=x.reshape(len(x),1)
y=y.reshape(len(y),1)

# %% [markdown] {"id":"S9nv9VkZ1EkG"}
# ### Training the Decision Tree Regression model on the whole dataset

# %% [code] {"id":"MIaU-C9a1EkG","jupyter":{"outputs_hidden":false}}
from sklearn.tree import DecisionTreeRegressor
regressor=DecisionTreeRegressor(random_state=0)
regressor.fit(x,y)

# %% [code] {"id":"3OzlCotZ1EkG","jupyter":{"outputs_hidden":false}}
DecisionTreeRegressor(random_state=0)

# %% [code] {"id":"FaRGTvzj1EkG","jupyter":{"outputs_hidden":false}}
## Predicting a new result: What is the price for the 
regressor.predict([[8042.21]])

# %% [code] {"id":"lpXWKZ3v1EkG","jupyter":{"outputs_hidden":false}}
regressor.predict([[8000]])

# %% [code] {"id":"13XBCzpY1EkG","jupyter":{"outputs_hidden":false}}
regressor.predict([[18000]])

# %% [markdown] {"id":"hXzFcwEV1EkG"}
# ### Visualization

# %% [code] {"id":"rlk3yMrU1EkG","jupyter":{"outputs_hidden":false}}
X_grid = np.arange(min(x), max(x), 0.001)
X_grid = X_grid.reshape((len(X_grid), 1))

plt.scatter(x,y, color="red")
plt.plot(X_grid, regressor.predict(X_grid), color="blue")

plt.title("Decision Tree Regression for Avocado Prices")
plt.xlabel("Avocado Small Bags")
plt.ylabel("Price")
plt.show()

# %% [markdown] {"id":"Y22RDjpA1EkH"}
# # Findings

# %% [markdown] {"id":"GAxS_2Aw1EkH"}
# - Conventional avocados sell way more than organic avocados and cost less. Therefore, Total volume, along with other volume variables, and average price, will work well to predict our target variable, type, in our classification model
# 
# - Average price and total volume move in different directions, this will come in handy when doing a regression analysis over our target variable, which is average price
# 
# - In the time series exploration, we see that there is a pike in total volume and a drop in prices at the beggining of the month, hinting for seasonality and forecasting possibilities

# %% [markdown] {"id":"50Weh7RH1EkH"}
# Normally, there is an inverse relationship between supply and prices. When there is an overproduction of avocados they will have a negative impact on the market price of avocados. Let's see if this is the case for both conventional and organic avocados.
# Conventional: At the end of 2017 we can see a large drop in prices, at the same time there is an increasing amount of volume of avocados in the market.
# Organic: Same happens with organic avocados, at the end of 2017 there is a big drop and we can see a huge increase in volume.
# Volume peaks: Notice how each volume peak is a signal for an upcoming drop in avocado prices.

# %% [code] {"id":"lFDMgBLr1EkH","jupyter":{"outputs_hidden":false}}
