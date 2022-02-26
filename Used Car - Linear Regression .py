#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
sns.set()


# In[2]:


df = pd.read_csv('/Users/josephthomas/Downloads/LinearRegressionCars.csv')


# In[3]:


df.head()


# In[6]:


df.describe(include='all')


# In[8]:


data = df.drop(['Model'],axis=1)
data.describe(include='all')


# In[10]:


data.isnull().sum()


# In[11]:


data_no_mv = data.dropna(axis=0)


# In[12]:


data_no_mv.describe(include='all')


# ### Exploring PDFs

# In[14]:


sns.displot(data_no_mv['Price'])


# In[15]:


#Dealing with Outliers 


# In[16]:


q = data_no_mv['Price'].quantile(0.99)
data_1 = data_no_mv[data_no_mv['Price']<q]
data_1.describe(include='all')


# In[17]:


sns.displot(data_1['Price'])


# In[18]:


sns.displot(data_no_mv['Mileage'])


# In[20]:


q = data_1['Mileage'].quantile(0.99)
data_2 = data_1[data_1['Mileage']<q]


# In[21]:


sns.displot(data_2['Mileage'])


# In[19]:


sns.displot(data_no_mv['EngineV'])


# In[22]:


#Following research regarding the natural domain of the engine volume values
data_3=data_2[data_2['EngineV']<6.5]


# In[23]:


sns.displot(data_3['EngineV'])


# In[24]:


sns.displot(data_no_mv['Year'])


# In[28]:


q = data_3['Year'].quantile(0.01)
data_4 = data_3[data_3['Year']>q]


# In[29]:


sns.displot(data_4['Year'])


# In[30]:


data_clean = data_4.reset_index(drop=True)


# In[31]:


data_clean.describe(include='all')


# ### Checking OLS Assumptions 

# In[39]:


f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize =(15,3))
ax1.scatter(data_clean['Year'],data_clean['Price'])
ax1.set_title('Price and Year')
ax2.scatter(data_clean['EngineV'],data_clean['Price'])
ax2.set_title('Price and EngineV')
ax3.scatter(data_clean['Mileage'],data_clean['Price'])
ax3.set_title('Price and Mileage')

plt.show()


# In[36]:


# PDF Plot showed Price to not be normally distributed


# In[37]:


log_price = np.log(data_clean['Price'])
data_clean['log_price'] = log_price
data_clean


# In[40]:


f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize =(15,3))
ax1.scatter(data_clean['Year'],data_clean['log_price'])
ax1.set_title('Log Price and Year')
ax2.scatter(data_clean['EngineV'],data_clean['log_price'])
ax2.set_title('Log Price and EngineV')
ax3.scatter(data_clean['Mileage'],data_clean['log_price'])
ax3.set_title('Log Price and Mileage')

plt.show()


# In[41]:


data_clean = data_clean.drop(['Price'],axis=1)


# In[42]:


#Checking for multicollinearity 


# In[43]:


data_clean.columns.values


# In[47]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
variables = data_clean[['Mileage','Year','EngineV']]
vif = pd.DataFrame()
vif['VIF'] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
vif['features']=variables.columns


# In[48]:


vif


# In[49]:


data_no_multico = data_clean.drop(['Year'], axis=1)


# In[50]:


#Creating Dummy Variables 


# In[52]:


data_with_dummies = pd.get_dummies(data_no_multico, drop_first=True)


# In[53]:


data_with_dummies.head()


# ### Rearange Columns 

# In[55]:


data_with_dummies.columns.values


# In[57]:


cols = ['log_price', 'Mileage', 'EngineV', 'Brand_BMW',
       'Brand_Mercedes-Benz', 'Brand_Mitsubishi', 'Brand_Renault',
       'Brand_Toyota', 'Brand_Volkswagen', 'Body_hatch', 'Body_other',
       'Body_sedan', 'Body_vagon', 'Body_van', 'Engine Type_Gas',
       'Engine Type_Other', 'Engine Type_Petrol', 'Registration_yes']


# In[59]:


data_prerocessed = data_with_dummies[cols]
data_prerocessed.head()


# ### Linear Regression Model 

# In[66]:


targets = data_prerocessed['log_price']
inputs = data_prerocessed.drop(['log_price'],axis=1)


# In[67]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(inputs)


# In[68]:


inputs_scaled = scaler.transform(inputs)


# #### Train Test Split 

# In[70]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(inputs_scaled, targets, test_size=0.2, random_state=42)


# #### Regression 

# In[71]:


reg = LinearRegression()
reg.fit(x_train,y_train)


# In[72]:


y_hat = reg.predict(x_train)


# In[73]:


plt.scatter(y_train, y_hat)
plt.xlabel('Targets (y_train)',size=18)
plt.ylabel('Predictions (y_hat)',size=18)
plt.xlim(6,13)
plt.ylim(6,13)
plt.show()


# In[75]:


sns.displot(y_train - y_hat)
plt.title('Residuals PDF', size=18)


# #### This plot shows that there are certain observations for which (y_train - y_hat) is much lower than the mean

# In[76]:


reg.score(x_train,y_train)


# ### Finding Weights and Bias 

# In[77]:


reg.intercept_


# In[79]:


reg.coef_


# In[81]:


reg_summary = pd.DataFrame(inputs.columns.values, columns=['Features'])
reg_summary['Weights'] = reg.coef_
reg_summary


# In[83]:


data_clean['Brand'].unique()


# In[84]:


data_clean['Body'].unique()


# In[85]:


data_clean['Engine Type'].unique()


# ### Testing 

# In[86]:


y_hat_test = reg.predict(x_test)


# In[91]:


plt.scatter(y_test, y_hat_test, alpha=0.2)
plt.xlabel('Targets (y_test)',size=18)
plt.ylabel('Predictions (y_hat_test)',size=18)
plt.xlim(6,13)
plt.ylim(6,13)
plt.show()


# In[101]:


df_pf = pd.DataFrame(np.exp(y_hat_test), columns=['Prediction'])
df_pf.head()


# In[102]:


df_pf['Target'] = np.exp(y_test)
df_pf


# In[96]:


y_test = y_test.reset_index(drop=True)
y_test.head()


# In[103]:


df_pf['Target'] = np.exp(y_test)
df_pf


# In[104]:


df_pf['Residual'] = df_pf['Target'] - df_pf['Prediction']


# In[105]:


df_pf['Difference%'] = np.absolute(df_pf['Residual']/df_pf['Target']*100)
df_pf


# In[106]:


df_pf.describe()


# In[108]:


pd.options.display.max_rows = 999
pd.set_option('display.float_format', lambda x: '%.2f' % x)
df_pf.sort_values(by=['Difference%'])

