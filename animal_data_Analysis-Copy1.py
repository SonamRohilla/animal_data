#!/usr/bin/env python
# coding: utf-8

# In[92]:


import numpy;


# In[93]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[94]:


animal_data = pd.read_csv('animal_speeds.csv', delimiter=';', header=0)


# Exploratory Data Analysis

# In[95]:


animal_data.info()


# In[96]:


animal_data.describe(include='all')


# In[97]:


animal_data.head()


# In[98]:


numerical_features = [feature for feature in animal_data.columns if data[feature].dtypes != 'O']
categorical_features = [feature for feature in animal_data.columns if data[feature].dtype == 'O']
print("numerical_features")
print(numerical_features)
print()
print('categorical_features')
print(categorical_features)


# # Missing Data we use seaborn heatmap to check whether we have null values or not

# In[99]:


animal_data.isnull()


# ###### Heatmap to visualise the null values

# In[100]:


sns.heatmap(animal_data.isnull(),yticklabels=False, cbar=False, cmap='viridis')


# ####### No parameters has null value

# In[101]:


sns.set_style('whitegrid')
sns.countplot(x='highspeed', data=animal_data)


# In[102]:


sns.set_style('whitegrid')
sns.countplot(x='weight', data=animal_data)


# In[103]:


sns.set_style('whitegrid')
sns.countplot(x='movement_type', data=animal_data)


# # around 78% animals' movement type is running, 41% is swimming and 28%, 13%  animals' movement type is flying and climbing respectively.

# In[104]:


sns.boxplot(x='movement_type', y='weight', data=animal_data, palette='winter')


# In[105]:


sns.boxplot(x='movement_type', y='highspeed', data=animal_data, palette='winter')


# In[106]:


for feature in categorical_features:
    sns.countplot(data=animal_data, x=feature)
    plt.show()


# In[107]:


for feature in numerical_features:
    sns.histplot(data=animal_data, x=feature,kde=True, bins=30, color='blue')
    plt.show();


# In[108]:


sns.countplot(x='highspeed', data= animal_data)


# In[109]:


sns.countplot(x='weight', data= animal_data)


# # converting categorial features

# In[111]:


movement=pd.get_dummies(animal_data['movement_type']).head()


# In[112]:


movement.head()


# In[113]:


animal_data1=pd.concat([animal_data,movement],axis=1)


# In[114]:


animal_data1.head()


# # Building a logistics regression model 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




