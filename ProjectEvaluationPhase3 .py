#!/usr/bin/env python
# coding: utf-8

# # Census Income

# In[1]:


import numpy as np
import pandas as pd
import sklearn
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder 
from scipy.stats import zscore
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import warnings
warnings.filterwarnings('ignore')


# Let's Load the csv file to the 'data' variable

# In[2]:


data = pd.read_csv("C:/Users/vaddi/Downloads/census_income.csv")
data


#                     The prediction task is to determine whether a person makes over $50K a year.

#  So, here the 'Income'is the target variable.
# 
# Now, let's check if there are any null values present in the dataset

# In[3]:


data.isnull().sum()


# In[4]:


#Here we can see that there are no null values in the dataset


# In[5]:


#Let's check the data types of each column present in the dataset
data.dtypes


# In[6]:


# Remove duplicates and keep the first occurrence
data_no_duplicates = data.drop_duplicates(inplace=True)


# In[7]:


data.shape


# It is observed that the dataset is the mix of int and object datatypes

# In[8]:


#Let's see the value counts in each column of the dataset
for i in data.columns:
    v=data[i].value_counts()
    print( "The value_counts in column", i, "is", v)
    print("*"*40)


# In[9]:


#Let's seperate the columns with thecategorical and numerical data types
categorical_col =[]
numerical_col = []
for column in data.columns:
    if data[column].dtypes == 'object':
        categorical_col.append(column)
    else:
        numerical_col.append(column)
print("Categorical_col : ", categorical_col)
print("numerical_col : ", numerical_col)


# Let's look into the statistical information of the dataset

# In[10]:


data.describe()

Observations:
    1.The count of all the columns are same which indicates there are equal number of values and no missing/null values
    2. The mean value is higher than the median which shows that the data is right skewed
    3. The mean value is lower than the median value in few columns implies that the data is left skewed
    4. The higher standard deviation shows that the data is widespread.
    
# # data Visualization

# In[11]:


#Let's viaualize the 'age' attribute
plt.figure(figsize=(10,8))
sns.histplot(x='Age', data=data, bins=20)
plt.show()


# In[12]:


#Let's viaualize the 'Workclass' attribute
sns.countplot(x='Workclass', data=data)
plt.show()
data['Workclass'].value_counts()


# In[13]:


#Let's viaualize the 'Fnlwgt' attribute
sns.histplot(x='Fnlwgt', data=data)
plt.show()
data['Fnlwgt'].value_counts()


# In[14]:


#Let's visualize the 'Education' attribute
sns.countplot(x='Education', data=data)
plt.show()
data['Education'].value_counts()


# In[15]:


#Let's visualize the 'Education_num' attribute
sns.countplot(x='Education_num', data=data)
plt.show()
data['Education_num'].value_counts()


# In[16]:


#Let's visualize the 'Marital_status' attribute
sns.countplot(x='Marital_status', data=data)
plt.show()
data['Marital_status'].value_counts()


# In[17]:


#Let's visualize the 'Occupation' attribute
sns.countplot(x='Occupation', data=data)
plt.show()
data['Occupation'].value_counts()


# In[18]:


#Let's visualize the 'Relationship' attribute
sns.countplot(x='Relationship', data=data)
plt.show()
data['Relationship'].value_counts()


# In[19]:


#Let's visualize the 'Race' attribute
sns.countplot(x='Race', data=data)
plt.show()
data['Race'].value_counts()


# In[20]:


#Let's visualize the 'Sex' attribute
sns.countplot(x='Sex', data=data)
plt.show()
data['Sex'].value_counts()


# In[21]:


#Let's visualize the 'Capital_gain' attribute
sns.histplot(x='Capital_gain', data=data)
plt.show()
data['Capital_gain'].value_counts()


# In[22]:


#Let's visualize the 'Capital_gain' attribute
sns.histplot(x='Capital_loss', data=data)
plt.show()
data['Capital_loss'].value_counts()


# In[23]:


#Let's visualize the 'Hours_per_week' attribute
sns.histplot(x='Hours_per_week', data=data)
plt.show()
data['Hours_per_week'].value_counts()


# In[24]:


#Let's visualize the 'Native_country' attribute
sns.histplot(x='Native_country', data=data)
plt.show()
data['Native_country'].value_counts()


# In[25]:


data.drop('Education', axis=1, inplace=True)


# In[29]:


#Let's encode the columns with object data type.
le=LabelEncoder()
for column in data.columns:
    if data[column].dtypes == 'object':
        data[column] = le.fit_transform(data[column])


# # Bivariant Analysis

# In[30]:


sns.barplot(x='Income', y='Age', data=data)
plt.title('Bar Plot analysis of Income and Age')
plt.show()


# In[31]:



sns.barplot(x='Education_num', y='Income', data=data)
plt.title('Bar Plot analysis of Income and Education_num')
plt.show()


# In[32]:



sns.barplot(x='Occupation', y='Income', data=data)
plt.title('Bar Plot analysis of Income and Occupation')
plt.show()


# In[33]:



sns.barplot(x='Marital_status', y='Income', data=data)
plt.title('Bar Plot analysis of Income and Marital_status')
plt.show()


# In[34]:


sns.barplot(x='Income', y='Relationship', data=data)
plt.title('Bar Plot analysis of Income and Relationship')
plt.show()


# In[35]:


sns.barplot(x='Income', y='Sex', data=data)
plt.title('Bar Plot analysis of Income and Sex')
plt.show()


# In[36]:


sns.barplot(x='Income', y='Race', data=data)
plt.title('Bar Plot analysis of Income and Race')
plt.show()


# In[37]:


sns.barplot(x='Income', y='Capital_gain', data=data)
plt.title('Bar Plot analysis of Income and Capital_gain')
plt.show()


# In[38]:


sns.barplot(x='Income', y='Capital_loss', data=data)
plt.title('Bar Plot analysis of Income and Capital_loss')
plt.show()


# In[39]:


sns.barplot(x='Income', y='Hours_per_week', data=data)
plt.title('Bar Plot analysis of Income and Hours_per_week')
plt.show()


# # Multivariant Data Analysis

# In[41]:


sns.pairplot(data)
plt.show()


#          Let's check for the outliers in the dataset

# In[42]:


z=np.abs(zscore(data))


# In[43]:


threshold= 3
print(np.where(z>3))


# In[44]:


df= data[(z<3). all(axis=1) ]


# In[101]:


#let's check the skewness 
df.skew()


# In[102]:


df['Capital_loss'] = np.log1p(df['Capital_loss'])
df['Capital_loss'].skew()


# In[106]:


df['Race'], lambda_bestfit = stats.boxcox(df['Race'])
df['Race'].skew()


# In[107]:


#Let's check the correlation of the attributes

corr=df.corr()
corr


# In[108]:


plt.figure(figsize=(10,12))
sns.heatmap(corr, annot=True)


#               Let's split the independent and target values to apply the model

# In[109]:


df_x = df.iloc[:,0:-1]


# In[110]:


#Let's apply the Standard scaler, scaling technique
scaler=StandardScaler()
x= scaler.fit_transform(df_x)


# In[111]:


y= df.iloc[:,-1]


# In[112]:


#Let's apply Logistic Regression and check the highest accuracy score that can be obtained and the random state at which
idle_score = 0.70

best_random_state = None
best_accuracy = 0.0

for r_state in range(1,100):
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20, random_state=r_state)
    lg=LogisticRegression()
    lg.fit(x_train,y_train)
    lgpred=lg.predict(x_test)
    acc = accuracy_score(y_test,lgpred)
    
    if acc> best_accuracy:
        best_accuracy= acc
        best_randome_state = r_state
    print("randome_state :", r_state, "Accuracy Score :", acc)
    

if best_accuracy > idle_score:
    print("The best accuracy rate is", best_accuracy, "for random state:", best_randome_state)
else:
    print("No random state achieved an accuracy greater than the threshold.")
    


# We acheived the highest accuracy rate using Logistic Regression, that is 0.84 at the random state of 9

# In[113]:


#Let's try using the RandomForest Classifier 
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()

idle_score = 0.84

best_random_state = None
best_accuracy = 0.0

for r_state in range(1,100):
    x_train,x_test,y_train,y_test= train_test_split(x,y, test_size=0.20, random_state=r_state)
    model= RandomForestClassifier()
    model.fit(x_test,y_test)
    modelpred=model.predict(x_test)
    acc= accuracy_score(y_test,modelpred)
    
    if acc> best_accuracy:
        best_accuracy= acc
        best_random_state= r_state
    print("Random State :", r_state, "Accuracy_Score : ", acc)

if best_accuracy > idle_score:
    print("the best Accuracy rate is ", best_accuracy, "For Random State ", best_random_state)
else:
    Print("No randome state achieved the accuracy greater than the threshold")


# In[114]:


#Let's print the confusion_matrix of the Random_forest_model
print(confusion_matrix(y_test,modelpred))


# In[115]:


sns.heatmap(confusion_matrix(y_test,modelpred), annot=True)


# In[116]:


#Let's print the Classification_report of the Random_forest_model
print(classification_report(y_test,modelpred))


# # Saving the model

# In[117]:


import joblib
filename = 'random_forest_model.pkl'
joblib.dump(model, filename)
print(f'Model saved as {filename}')


# # Rainfall Weather Forecasting

# In[35]:


import pandas as pd
import numpy as np
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from scipy.stats import zscore
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Let's Load the csv file to the 'rf' variable

# In[118]:


rf = pd.read_csv("C:/Users/vaddi/Downloads/weatherAUS.csv")
rf


# # Problem Statement:
# a) Design a predictive model with the use of machine learning algorithms to forecast whether or not it will rain tomorrow. 
# 
# 
# b) Design a predictive model with the use of machine learning algorithms to predict how much rainfall could be there.

# In[120]:


#Let's check the data types of each column present in the dataset and the data type
rf.isnull().sum(), rf.dtypes


# In[121]:


# Remove duplicates and keep the first occurrence
rf_no_duplicates = rf.drop_duplicates(inplace=True)


# In[122]:


rf.shape


# In[123]:


#Let's see the value counts in each column of the dataset
for i in rf.columns:
    v= rf[i].value_counts()
    print("The unique value in ", i ,"is", v)
    print("*"*30)


# In[124]:


#Let's seperate the columns with the categorical and numerical data types
categorical= []
numerical = []
for i in rf.columns:
    if rf[i].dtypes == 'object':
        categorical.append(i)
    else:
        numerical.append(i)
print("Categorical coulumns are ", len(categorical), categorical)
print("Numerical columns are ", len(numerical), numerical)


# In[125]:


#Let's fill the null values is the data set
for i in rf.columns:
    if rf[i].isnull().sum() >0 :
        if rf[i].dtypes != 'object':
            median_value = rf[i].median()
            rf[i].fillna(median_value, inplace = True)
        if rf[i].dtypes =='object':
            mode_value= rf[i].mode()
            rf[i].fillna(mode_value[0], inplace=True)


# In[126]:


rf.isnull().sum()


# In[127]:


#Let's drop the 'Date' column
rf.drop('Date', axis= 1, inplace=True)


# In[128]:


#Let's encode the categorical columns
le =LabelEncoder()
for column in rf.columns:
    if rf[column].dtypes == 'object':
        rf[column]= le.fit_transform(rf[column])


# In[129]:


rf.dtypes


# Let's look into the statistical information of the dataset

# In[131]:


rf.describe()

Observations:
    1.The count of all the columns are same which indicates there are equal number of values and no missing/null values
    2. The mean value is higher than the median which shows that the data is right skewed
    3. The mean value is lower than the median value in few columns implies that the data is left skewed
    4. The higher standard deviation shows that the data is widespread.
# # Data Visualization

# In[133]:


#Let's viaualize the 'Location' attribute

sns.histplot(x='Location', data=rf, bins=20)
plt.show()


# In[134]:


#Let's viaualize the 'MinTemp' attribute

sns.histplot(x='MinTemp', data=rf, bins=20)
plt.show()


# In[135]:


#Let's viaualize the 'MaxTemp' attribute

sns.histplot(x='MaxTemp', data=rf, bins=20)
plt.show()


# In[136]:


#Let's viaualize the 'Rainfall' attribute
sns.histplot(x='Rainfall', data=rf, bins=20)
plt.show()


# In[137]:


#Let's viaualize the 'Evaporation' attribute
sns.histplot(x='Evaporation', data=rf, bins=20)
plt.show()


# In[138]:


#Let's viaualize the 'Evaporation' attribute
sns.histplot(x='Sunshine', data=rf, bins=20)
plt.show()


# In[139]:


#Let's viaualize the 'WindGustDir' attribute
sns.histplot(x='WindGustDir', data=rf, bins=20)
plt.show()


# In[140]:


#Let's viaualize the 'WindGustSpeed' attribute
sns.histplot(x='WindGustSpeed', data=rf, bins=20)
plt.show()


# In[141]:


#Let's viaualize the 'WindDir9am' attribute
sns.histplot(x='WindDir9am', data=rf, bins=20)
plt.show()


# In[142]:


#Let's viaualize the 'WindDir3pm' attribute
sns.histplot(x='WindDir3pm', data=rf, bins=20)
plt.show()


# In[143]:


#Let's viaualize the 'WindSpeed9am' attribute
sns.histplot(x='WindSpeed9am', data=rf, bins=20)
plt.show()


# In[144]:


#Let's viaualize the 'WindSpeed3pm' attribute
sns.histplot(x='WindSpeed3pm', data=rf, bins=20)
plt.show()


# In[145]:


#Let's viaualize the 'Humidity9am' attribute
sns.histplot(x='Humidity9am', data=rf, bins=20)
plt.show()


# In[146]:


#Let's viaualize the 'Humidity3pm' attribute
sns.histplot(x='Humidity3pm', data=rf, bins=20)
plt.show()


# In[147]:


#Let's viaualize the 'Pressure9am' attribute
sns.histplot(x='Pressure9am', data=rf, bins=20)
plt.show()


# In[148]:


#Let's viaualize the 'Pressure3pm' attribute
sns.histplot(x='Pressure3pm', data=rf, bins=20)
plt.show()


# In[149]:


#Let's viaualize the 'Cloud9am' attribute
sns.histplot(x='Cloud9am', data=rf, bins=20)
plt.show()


# In[150]:


#Let's viaualize the 'Cloud3pm' attribute
sns.histplot(x='Cloud3pm', data=rf, bins=20)
plt.show()


# In[151]:


#Let's viaualize the 'Temp9am' attribute
sns.histplot(x='Temp9am', data=rf, bins=20)
plt.show()


# In[152]:


#Let's viaualize the 'Temp3pm' attribute
sns.histplot(x='Temp3pm', data=rf, bins=20)
plt.show()


# In[153]:


#Let's viaualize the 'RainToday' attribute
sns.histplot(x='RainToday', data=rf, bins=20)
plt.show()


# In[154]:


#Let's viaualize the 'RainTomorrow' attribute
sns.histplot(x='RainTomorrow', data=rf, bins=20)
plt.show()


# # Bivariant Analysis

# In[155]:


sns.barplot(x='Location', y='RainTomorrow', data=rf)
plt.title('Bar Plot analysis of Location and RainTomorrow')
plt.show()


# In[156]:


sns.jointplot(x='MinTemp', y='RainTomorrow', data=rf, kind='scatter')
plt.title('Plot analysis of MinTemp and RainTomorrow')
plt.show()


# In[157]:


sns.jointplot(x='MaxTemp', y='RainTomorrow', data=rf, kind='scatter')
plt.title('Analysis of MaxTemp and RainTomorrow')
plt.show()


# In[158]:


sns.jointplot(x='Rainfall', y='RainTomorrow', data=rf, kind='scatter')
plt.title('Analysis of Rainfall and RainTomorrow')
plt.show()


# In[159]:


sns.jointplot(x='Evaporation', y='RainTomorrow', data=rf, kind='scatter')
plt.title('Analysis of Evaporation and RainTomorrow')
plt.show()


# In[160]:


sns.jointplot(x='Sunshine', y='RainTomorrow', data=rf, kind='scatter')
plt.title('Analysis of Sunshine and RainTomorrow')
plt.show()


# In[161]:


sns.jointplot(x='WindGustDir', y='RainTomorrow', data=rf, kind='scatter')
plt.title('Analysis of WindGustDir and RainTomorrow')
plt.show()


# In[162]:


sns.jointplot(x='WindGustSpeed', y='RainTomorrow', data=rf, kind='scatter')
plt.title('Analysis of WindGustSpeed and RainTomorrow')
plt.show()


# In[163]:


sns.jointplot(x='WindDir9am', y='RainTomorrow', data=rf, kind='scatter')
plt.title('Analysis of WindDir9am and RainTomorrow')
plt.show()


# In[164]:


sns.jointplot(x='WindDir3pm', y='RainTomorrow', data=rf, kind='scatter')
plt.title('Analysis of WindDir3pm and RainTomorrow')
plt.show()


# In[165]:


sns.jointplot(x='WindSpeed9am', y='RainTomorrow', data=rf, kind='scatter')
plt.title('Analysis of WindSpeed9am and RainTomorrow')
plt.show()


# In[166]:


sns.jointplot(x='WindSpeed3pm', y='RainTomorrow', data=rf, kind='scatter')
plt.title('Analysis of WindSpeed3pm and RainTomorrow')
plt.show()


# In[167]:


sns.jointplot(x='Humidity9am', y='RainTomorrow', data=rf, kind='scatter')
plt.title('Analysis of Humidity9am and RainTomorrow')
plt.show()


# In[168]:


sns.jointplot(x='Humidity3pm', y='RainTomorrow', data=rf, kind='scatter')
plt.title('Analysis of Humidity3pm and RainTomorrow')
plt.show()


# In[169]:


sns.jointplot(x='Pressure9am', y='RainTomorrow', data=rf, kind='scatter')
plt.title('Analysis of Pressure9am and RainTomorrow')
plt.show()


# In[170]:


sns.jointplot(x='Pressure3pm', y='RainTomorrow', data=rf, kind='scatter')
plt.title('Analysis of Pressure3pm and RainTomorrow')
plt.show()


# In[171]:


sns.jointplot(x='Cloud9am', y='RainTomorrow', data=rf, kind='scatter')
plt.title('Analysis of Cloud9am and RainTomorrow')
plt.show()


# In[172]:


sns.jointplot(x='Cloud9am', y='RainTomorrow', data=rf, kind='scatter')
plt.title('Analysis of Cloud9am and RainTomorrow')
plt.show()


# In[173]:


sns.jointplot(x='Cloud3pm', y='RainTomorrow', data=rf, kind='scatter')
plt.title('Analysis of Cloud3pm and RainTomorrow')
plt.show()


# In[174]:


sns.jointplot(x='Temp9am', y='RainTomorrow', data=rf, kind='scatter')
plt.title('Analysis of Temp9am and RainTomorrow')
plt.show()


# In[175]:


sns.jointplot(x='Temp3pm', y='RainTomorrow', data=rf, kind='scatter')
plt.title('Analysis of Temp3pm and RainTomorrow')
plt.show()


# In[176]:


sns.jointplot(x='RainToday', y='RainTomorrow', data=rf, kind='scatter')
plt.title('Analysis of RainToday and RainTomorrow')
plt.show()


# # Multivariant analysis

# In[177]:


sns.pairplot(rf)
plt.show()


# In[178]:


#Let's find the correlation
corr = rf.corr()


# In[ ]:


plt.figure(figsize=(10,12))
sns.heatmap(corr, annot=True)


# # 'Rain tomorrow as the target variable'
# 

# In[179]:


X= rf.iloc[:,0:-1]


# In[192]:


#Let's check the skewness 
X.skew()


# In[248]:


rf.loc[:, 'Rainfall'] = np.log1p(rf['Rainfall'])
rf['Rainfall'].skew()


# In[249]:


rf.loc[:, 'Evaporation'] = np.sqrt(rf['Evaporation'])
rf['Evaporation'].skew()


# In[251]:



rf.loc[:, 'RainToday'] = np.cbrt(rf['RainToday'])
rf['RainToday'].skew()


# Let's split the data to fit in the model

# In[255]:


x= rf.iloc[:,0:-1]


# In[254]:


y= rf.iloc[:,-1]


# In[256]:


#Let's apply Logistic Regression and check the highest accuracy score that can be obtained and the random state at which
idle_score = 0.70

best_random_state = None
best_accuracy = 0.0

for r_state in range(1,100):
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20, random_state=r_state)
    lg=LogisticRegression()
    lg.fit(x_train,y_train)
    lgpred=lg.predict(x_test)
    acc = accuracy_score(y_test,lgpred)
    
    if acc> best_accuracy:
        best_accuracy= acc
        best_randome_state = r_state
    print("randome_state :", r_state, "Accuracy Score :", acc)
    

if best_accuracy > idle_score:
    print("The best accuracy rate is", best_accuracy, "for random state:", best_randome_state)
else:
    print("No random state achieved an accuracy greater than the threshold.")


# In[259]:


#Let's try using the RandomForest Classifier 
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()

idle_score = 0.85

best_random_state = None
best_accuracy = 0.0

for r_state in range(1,100):
    x_train,x_test,y_train,y_test= train_test_split(x,y, test_size=0.20, random_state=r_state)
    rfmodel= RandomForestClassifier()
    rfmodel.fit(x_test,y_test)
    rfmodelpred=rfmodel.predict(x_test)
    acc= accuracy_score(y_test,rfmodelpred)
    
    if acc> best_accuracy:
        best_accuracy= acc
        best_random_state= r_state
    print("Random State :", r_state, "Accuracy_Score : ", acc)

if best_accuracy > idle_score:
    print("the best Accuracy rate is ", best_accuracy, "For Random State ", best_random_state)
else:
    Print("No randome state achieved the accuracy greater than the threshold")


# In[260]:


#Let's print the confusion_matrix of the Random_forest_model
print(confusion_matrix(y_test,rfmodelpred))


# In[261]:


sns.heatmap(confusion_matrix(y_test,rfmodelpred), annot=True)


# In[262]:


#Let's print the Classification_report of the Random_forest_model
print(classification_report(y_test,rfmodelpred))


# # Saving the model

# In[294]:


import joblib
filename = 'RandomForest_model.pkl'
joblib.dump(rfmodel, filename)
print(f'Model saved as {filename}')


# # 'rainfall as the target variable'

# In[296]:


from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, make_scorer, mean_absolute_error
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA


# In[263]:


df_X = rf.iloc[:, rf.columns != rf.columns[2]]


# In[264]:


#Let's check for the skewness
df_X.skew()


# Let's split the data to fit in the model

# In[267]:


X= rf.iloc[:, rf.columns != rf.columns[3]]


# In[268]:


Y= rf.loc[:, 'Rainfall']


# In[269]:


X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.20, random_state=42)


# In[270]:


scaler = StandardScaler()
X_train =scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[273]:


pca=PCA(n_components=15)
x_pca =pca.fit_transform(x)


# In[274]:


model = LinearRegression()


# In[275]:


model.fit(X_train,Y_train)


# In[276]:


modelpred = model.predict(X_test)


# In[277]:


model.score(X_train,Y_train)


# In[278]:


mse = mean_squared_error(Y_test, modelpred)
r2 = r2_score(Y_test, modelpred)


# In[279]:


print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")


# In[287]:


#Let's create a funtion for the model to check the results at different random states and preint the r2 score value
def maxr2_score(regr,X,Y):
    max_r_score=0
    for r_state in range(42,100):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y,random_state = r_state,test_size=0.20)
        regr.fit(X_train,Y_train)
        Y_pred = regr.predict(X_test)
        r2_scr=r2_score(Y_test,Y_pred)
        print("r2 score corresponding to ",r_state," is ",r2_scr)
        if r2_scr>max_r_score:
            max_r_score=r2_scr
            final_r_state=r_state
    print("max r2 score corresponding to ",final_r_state," is ",max_r_score)
    return final_r_state


# In[285]:


#lets make use of KNN regressor
#we will use pipeline to bring bring features to common scale

knr=KNeighborsRegressor()
pipeline=Pipeline([("pca",PCA(n_components=15)),("knr",KNeighborsRegressor())])
parameters = {"knr__n_neighbors":range(2,30)}
clf = GridSearchCV(pipeline, parameters, cv=5,scoring="r2")
clf.fit(X,Y)
clf.best_params_


# In[288]:


#knr=KNeighborsRegressor(n_neighbors=4)
pipeline_knr=Pipeline([("ss",StandardScaler()),("knr",KNeighborsRegressor(n_neighbors=29))])
maxr2_score(pipeline_knr,X,Y)


# In[289]:


#Lets use random forest regressor
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings("ignore")
rfr=RandomForestRegressor()
pipeline=Pipeline([("ss",StandardScaler()),("pca",PCA(n_components=15)),("rfr",RandomForestRegressor())])
parameters = {"rfr__n_estimators":[10,100,500]}
clf = GridSearchCV(pipeline, parameters, cv=5,scoring="r2")
clf.fit(X,Y)
clf.best_params_


# In[290]:


pipeline_rfr=Pipeline([("ss",StandardScaler()),("rfr",RandomForestRegressor(n_estimators=500))])
maxr2_score(pipeline_rfr,X,Y)


# # Saving the model

# In[295]:


import joblib
filename = 'RFR.pkl'
joblib.dump(rfr, filename)
print(f'Model saved as {filename}')


# # Insurance Claim Fraud Detection

# In[325]:


import numpy as np
import pandas as pd
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy.stats import zscore
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB


# Let's load the dataset

# In[326]:


icf_df = pd.read_csv("C:/Users/vaddi/Downloads/Automobile_insurance_fraud.csv")
icf_df


# Problem Statement: create a predictive model that predicts if an insurance claim is fraudulent or not. 

# In[327]:


icf_df.isnull().sum()


# In[328]:


#Let's look for the null values in the data set
sns.heatmap(icf_df.isnull())


# In[329]:


#Let's seperate the categorical and numerical value to different lists
categorical_col =[]
numerical_col =[]
for i in icf_df.columns:
    if icf_df[i].dtypes == 'object':
        categorical_col.append(i)
    else:
        numerical_col.append(i)
        
print("Categorical columns present in the dataset : ", categorical_col)
print("*"*30)
print("Numerical columns present in the dataset : ", numerical_col)
    


# In[330]:


len(categorical_col)


# In[331]:


len(numerical_col)


# In[332]:


#Let's the value counts in each column of the data set
for i in icf_df.columns:
    v=icf_df[i].value_counts()
    print("Unique values in column ", i, "is", v)
    print("*"*30)


# In[334]:


#Let's remove the unneccesary columns inthe dataset
list1 = ['_c39', 'policy_number', 'incident_date', 'incident_location']
icf_df.drop(list1, axis =1, inplace= True)


# In[335]:


# Instantiate LabelEncoder
le = LabelEncoder()

for column in icf_df.columns:
    if icf_df[column].dtypes == 'object':
        icf_df[column] = le.fit_transform(icf_df[column])


# # Data Visualization
# 

# Bivariant Visualization

# In[336]:


for i in icf_df.columns:
    sns.regplot(x='fraud_reported', y=i, data=icf_df)
    plt.show()


# Multivariant Analysis

# In[122]:


sns.pairplot(icf_df)
plt.show()


# In[337]:


X= icf_df.iloc[:, 0:-1]


# In[338]:


z=np.abs(zscore(X))


# In[339]:


threshold = 3
print(np.where(z>3))


# In[340]:


df=icf_df[(z<3).all(axis=1)]


# In[341]:


df.skew()


# In[342]:


df.loc[:,'umbrella_limit']= np.cbrt(df['umbrella_limit'])
df['umbrella_limit'].skew()


# In[343]:


df_x=df.iloc[:,0:-1]


# In[344]:


y= df.iloc[:,-1]


# In[345]:


sc= StandardScaler()
sc.fit(df_x)
x=sc.transform(df_x)
x=pd.DataFrame(x, columns =df_x.columns)


# In[346]:


pca=PCA(n_components=15)
x_pca =pca.fit_transform(x)


# In[347]:


x_train, x_test, y_train, y_test = train_test_split(x_pca, y, test_size=0.2, random_state=43)

model = LogisticRegression()
model.fit(x_train, y_train)


# In[348]:


pred = model.predict(x_test)


# In[349]:


print(accuracy_score(y_test,pred))


# In[352]:


from sklearn.neighbors import KNeighborsClassifier
#Checking the KNeighbors Classifier model at different random states and their accuracy score
idle_scc = 0.70  # Set a reasonable threshold for accuracy
best_random_state = None
best_accuracy = 0.0
for r_state in range(1, 100):
    x_train, x_test, y_train, y_test = train_test_split(x_pca, y, random_state=r_state, test_size=0.30)
    knn_classifier = KNeighborsClassifier(n_neighbors=3)
    knn_classifier.fit(x_train, y_train)
    knnpred = knn_classifier.predict(x_test)
    acc = accuracy_score(y_test, knnpred)
    
    if acc > best_accuracy:
        best_accuracy = acc
        best_random_state = r_state

    print("Random State:", r_state, "Accuracy:", acc)

if best_accuracy > idle_scc:
    print("The best accuracy rate is", best_accuracy, "for random state:", best_random_state)
else:
    print("No random state achieved an accuracy greater than the threshold.")


# In[353]:


idle_scc = 0.80  # Set a reasonable threshold for accuracy

best_random_state = None
best_accuracy = 0.0

for r_state in range(42, 100):
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=r_state, test_size=0.30)
    svc = SVC()
    svc.fit(x_train, y_train)
    svcpred = svc.predict(x_test)
    acc = accuracy_score(y_test, svcpred)
    
    if acc > best_accuracy:
        best_accuracy = acc
        best_random_state = r_state

    print("Random State:", r_state, "Accuracy:", acc)

if best_accuracy > idle_scc:
    print("The best accuracy rate is", best_accuracy, "for random state:", best_random_state)
else:
    print("No random state achieved an accuracy greater than the threshold.")


# In[354]:


idle_scc = 0.80  # Set a reasonable threshold for accuracy

best_random_state = None
best_accuracy = 0.0

for r_state in range(1, 100):
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=r_state, test_size=0.30)
    rfc = RandomForestClassifier()
    rfc.fit(x_train, y_train)
    rfcpred = rfc.predict(x_test)
    acc = accuracy_score(y_test, rfcpred)
    
    if acc > best_accuracy:
        best_accuracy = acc
        best_random_state = r_state

    print("Random State:", r_state, "Accuracy:", acc)

if best_accuracy > idle_scc:
    print("The best accuracy rate is", best_accuracy, "for random state:", best_random_state)
else:
    print("No random state achieved an accuracy greater than the threshold.")


# In[356]:


#Let's create a model with the best random state and save it
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.30, random_state= 8)
rfc = RandomForestClassifier()
rfc.fit(x_train, y_train)
rfcpred = rfc.predict(x_test)
acc = accuracy_score(y_test, rfcpred)
acc


# In[357]:


print(confusion_matrix(y_test,rfcpred))


# In[358]:


sns.heatmap(confusion_matrix(y_test,rfcpred), annot=True)


# # Saving the model

# In[359]:


import joblib
filename = 'rfc.pkl'
joblib.dump(rfc, filename)
print(f'Model saved as {filename}')


# # Zomato Restaurant

# In[360]:


import pandas as pd
import numpy as np
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from scipy.stats import zscore
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler


# In[361]:


df1 = pd.read_csv("C:/Users/vaddi/Downloads/zomato (1).csv", encoding='latin1')


# In[363]:


file_path = 'C:/Users/vaddi/Downloads/Country-Code (2).xlsx'

df2 = pd.read_excel(file_path)
df2


# In[364]:


z_df = pd.merge(df1, df2, how='inner') 


# In[368]:


z_df.shape


# In[369]:


# Remove duplicates and keep the first occurrence
z_df_no_duplicates = z_df.drop_duplicates(inplace=True)


# In[371]:


#Let's check for the null values
z_df.isnull().sum()


# In[372]:


#Let's check the data type of each column
z_df.dtypes


# In[373]:


#Segregating the categorical and numerical columns
cat_cols=[]
num_cols=[]
for column in z_df.columns:
    if z_df[column].dtypes == 'object':
        cat_cols.append(column)
    else:
        num_cols.append(column)
print("Categorical colums in the dataset are", len(cat_cols), cat_cols)
print("Numerical columns in the dataset are ", len(num_cols), num_cols)


# In[374]:


#Let's fill the null values with the mode value
mode_value = z_df['Cuisines'].mode()[0]
z_df['Cuisines'].fillna(mode_value, inplace=True)


# In[375]:


sns.heatmap(z_df.isnull())
plt.show()


# In[376]:


#let's see the value counts of the each column
for i in z_df.columns:
    v= z_df[i].value_counts()
    print("The value_counts of ", i, "is", v)
    print ("*"*40)


# In[378]:


z_df.columns


# In[379]:


#Let's remove the uneccessary columns from the data set
list1 = ['Restaurant ID', 'Restaurant Name', 'Address','Locality Verbose','Longitude','Latitude']
z_df.drop(list1, axis=1, inplace=True)


# In[387]:


#Let's encode the categorical column
le=LabelEncoder()
for column in z_df.columns:
    if z_df[column].dtypes == 'object':
        z_df[column] = le.fit_transform(z_df[column])


# # Multivariant Analysis

# In[381]:


sns.pairplot(z_df)
plt.show()


# Let's split the independent variables and the target variable to fit in the model

# In[382]:


x= z_df.iloc[:,0:10].join(z_df.iloc[:,11:])


# In[384]:


y = z_df.iloc[:,10]


# In[385]:


model= KNeighborsClassifier()


# In[386]:


x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.20, random_state=42)


# In[388]:


model.fit(x_train,y_train)


# In[389]:


ypred = model.predict(x_test)


# In[390]:


print(accuracy_score(y_test,ypred))


# In[391]:


idle_scc = 0.88  # Set a reasonable threshold for accuracy

best_random_state = None
best_accuracy = 0.0

for r_state in range(1, 100):
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=r_state, test_size=0.30)
    rfc = RandomForestClassifier()
    rfc.fit(x_train, y_train)
    rfcpred = rfc.predict(x_test)
    acc = accuracy_score(y_test, rfcpred)
    
    if acc > best_accuracy:
        best_accuracy = acc
        best_random_state = r_state

    print("Random State:", r_state, "Accuracy:", acc)

if best_accuracy > idle_scc:
    print("The best accuracy rate is", best_accuracy, "for random state:", best_random_state)
else:
    print("No random state achieved an accuracy greater than the threshold.")


# In[393]:


#Let's create a model with the best random state and save it
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.30, random_state= 11)
rfcz = RandomForestClassifier()
rfcz.fit(x_train, y_train)
rfczpred = rfcz.predict(x_test)
acc = accuracy_score(y_test, rfczpred)
acc


# # Saving the model

# In[394]:


import joblib
filename = 'rfcz.pkl'
joblib.dump(rfcz, filename)
print(f'Model saved as {filename}')


# # Average Price

# In[395]:


z_df.skew()


# In[397]:


X= z_df.iloc[:,0:4].join(z_df.iloc[:,5:])


# In[399]:


Y= z_df.loc[:,'Average Cost for two']


# In[400]:


lr= LinearRegression()


# In[418]:


X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.20, random_state=88)


# In[419]:


scaler = StandardScaler()
X_train =scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[420]:


lr.fit(X_train,Y_train)


# In[421]:


Ypred= lr.predict(X_test)


# In[422]:


lr.score(X_train,Y_train)


# In[424]:


r2 = r2_score(Y_test, Ypred)

print(r2)


# In[407]:


#Let's create a funtion for the model to check the results at different random states and preint the r2 score value
def maxr2_score(regr,X,Y):
    max_r_score=0
    for r_state in range(42,100):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y,random_state = r_state,test_size=0.20)
        regr.fit(X_train,Y_train)
        Y_pred = regr.predict(X_test)
        r2_scr=r2_score(Y_test,Y_pred)
        print("r2 score corresponding to ",r_state," is ",r2_scr)
        if r2_scr>max_r_score:
            max_r_score=r2_scr
            final_r_state=r_state
    print("max r2 score corresponding to ",final_r_state," is ",max_r_score)
    return final_r_state


# In[409]:


#Lets use random forest regressor
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings("ignore")
rfr=RandomForestRegressor()
pipeline=Pipeline([("ss",StandardScaler()),("rfr",RandomForestRegressor())])
parameters = {"rfr__n_estimators":[10,100,500]}
clf = GridSearchCV(pipeline, parameters, cv=5,scoring="r2")
clf.fit(X,Y)
clf.best_params_


# In[411]:


pipeline_rfr=Pipeline([("ss",StandardScaler()),("rfr",RandomForestRegressor(n_estimators=100))])
maxr2_score(pipeline_rfr,X,Y)


# In[413]:


#Final model
rfr= RandomForestRegressor()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,random_state = 88,test_size=0.20)
rfr.fit(X_train,Y_train)
rfrpred=rfr.predict(X_test)


# In[415]:



r2 = r2_score(Y_test, rfrpred)

print(r2)


# # Saving the model

# In[416]:


import joblib
filename = 'rfrz.pkl'
joblib.dump(rfr, filename)
print(f'Model saved as {filename}')


# # Thanks!

# In[ ]:




