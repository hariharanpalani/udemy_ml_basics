# coding: utf-8

# # Multiple Linear Regression Demo to students

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


#data import
df=pd.read_csv(r'<put your file here>\auto-mpg.data',header=None,delimiter='\s+', names=['mpg','cylinders','displacement','horsepower','weight','acceleration','mode year','origin','car name'])
df.head()


# ##Data Exploration

# In[3]:


df.shape


# In[4]:


df=df.drop(df[df.horsepower=="?"].index)
df.head()


# In[5]:


df.shape


# In[6]:


df.applymap(np.isreal)


# In[7]:


df["horsepower"]=pd.to_numeric(df["horsepower"])


# In[8]:


df.applymap(np.isreal)


# In[9]:


df.isnull().sum()


# In[10]:


df.corr()


# In[11]:


#Variance Inflation Factor (VIF) calculation
from statsmodels.stats.outliers_influence import variance_inflation_factor
df1=df._get_numeric_data()
X=df1.drop(["mpg","origin"],axis=1)

vif=pd.DataFrame()
vif["VIF Factor"]=[variance_inflation_factor(X.values,i) for i in range(X.shape[1])]
vif["features"]=X.columns
vif.round(1)


# In[12]:


#Variance Inflation Factor (VIF) calculation
from statsmodels.stats.outliers_influence import variance_inflation_factor
df1=df._get_numeric_data()
X=df1.drop(["mpg","origin","weight"],axis=1)

vif=pd.DataFrame()
vif["VIF Factor"]=[variance_inflation_factor(X.values,i) for i in range(X.shape[1])]
vif["features"]=X.columns
vif.round(1)


# In[13]:


#Variance Inflation Factor (VIF) calculation
from statsmodels.stats.outliers_influence import variance_inflation_factor
df1=df._get_numeric_data()
X=df1.drop(["mpg","origin","weight","cylinders"],axis=1)

vif=pd.DataFrame()
vif["VIF Factor"]=[variance_inflation_factor(X.values,i) for i in range(X.shape[1])]
vif["features"]=X.columns
vif.round(1)


# In[14]:


#Variance Inflation Factor (VIF) calculation
from statsmodels.stats.outliers_influence import variance_inflation_factor
df1=df._get_numeric_data()
X=df1.drop(["mpg","origin","weight","cylinders","mode year"],axis=1)

vif=pd.DataFrame()
vif["VIF Factor"]=[variance_inflation_factor(X.values,i) for i in range(X.shape[1])]
vif["features"]=X.columns
vif.round(1)


# In[15]:


#Variance Inflation Factor (VIF) calculation
from statsmodels.stats.outliers_influence import variance_inflation_factor
df1=df._get_numeric_data()
X=df1.drop(["mpg","origin","weight","cylinders","mode year","horsepower"],axis=1)

vif=pd.DataFrame()
vif["VIF Factor"]=[variance_inflation_factor(X.values,i) for i in range(X.shape[1])]
vif["features"]=X.columns
vif.round(1)


# In[16]:


sns.pairplot(df1,x_vars=["displacement","acceleration"], y_vars="mpg",size=7.0)
plt.show()


# In[17]:


#Backward Elimination
import statsmodels.api as sm
#y=b0+b1X1+b2X2
#"mpg"=b0+b1*"displacement"+b2*"acceleration"
#"mpg"=b0*1+b1*"displacement"+b2*"acceleration"
y=df["mpg"]
X=sm.add_constant(X)
#X.head()
regressorOLS=sm.OLS(y,X).fit()
regressorOLS.summary()


# In[18]:


X=X.drop(["acceleration"],axis=1)
regressorOLS=sm.OLS(y,X).fit()
regressorOLS.summary()


# In[19]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)


# In[22]:


#train the model
from sklearn.linear_model import LinearRegression
linear_reg=LinearRegression()
linear_reg.fit(X_train,y_train)


# In[25]:


#predict y
y_pred=linear_reg.predict(X_test)


# In[26]:


df1 = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df1.head(15)


# In[30]:


sns.distplot(y_test,color="r",label="Actual Values",hist=False)
sns.distplot(y_pred,color="g",label="Predicted Values",hist=False)
plt.show()

