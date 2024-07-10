#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
get_ipython().system('pip install tensorflow')


# In[ ]:





# In[2]:


df=pd.read_csv(r'D:\study\Telecom_customer_churn\Telco-Customer-Churn.csv')


# In[3]:


df.sample(5)


# In[4]:


df.drop('customerID',axis='columns',inplace=True)
df.dtypes


# In[5]:


df.TotalCharges.values


# In[6]:


df.MonthlyCharges.values


# In[7]:


#changing TotalCharge column data type from object to float


# In[8]:


pd.to_numeric(df.TotalCharges,errors='coerce')#errors='coerce': Any value that cannot be converted to a numeric type will be replaced with NaN (Not a Number). This is useful when dealing with columns that might have some invalid entries that you want to ignore or handle separately.


# In[9]:


df[pd.to_numeric(df.TotalCharges,errors='coerce').isnull()]


# In[10]:


df.shape


# In[11]:


df.iloc[488]['TotalCharges']#just looking at sepecific rows and colomns


# In[12]:


df1 = df[df['TotalCharges'].apply(pd.to_numeric, errors='coerce').notnull()]
df1.shape


# In[13]:


df1['TotalCharges']=df1['TotalCharges'].astype(float) 
df1.dtypes


# # EDA

# In[14]:


ax=sns.countplot(x='Churn',data=df1)
for bars in ax.containers:
    ax.bar_label(bars)


# In[15]:


grouped_df1=df1.groupby(['Churn'],as_index=False).size()


# In[16]:


print(grouped_df1)


# In[17]:


grouped_df1.rename(columns={'size':'Count'})


# In[18]:


tenure_churn_no=df1[df1.Churn=='No'].tenure
tenure_churn_yes=df1[df1.Churn=='Yes'].tenure


# In[19]:


plt.hist([tenure_churn_yes,tenure_churn_no],color=['green','red'],label=['Churn=Yes','Churn=No'])
plt.legend()
plt.xlabel("tenure")
plt.ylabel("number of customers")
plt.title('Customer churn prediction visualization')
plt.show()


# In[20]:


mc_churn_no=df1[df1.Churn=='No'].MonthlyCharges
mc_churn_yes=df1[df1.Churn=='Yes'].MonthlyCharges
plt.figure(figsize=(10, 6))
plt.hist([mc_churn_yes,mc_churn_no],color=['green','red'],label=['Churn=yes','Churn=No'])
plt.xlabel='Monthly Charges'
plt.ylabel='Number of Customers'
plt.title='Customer Churn by Monthly Charges'


# In[21]:


def print_unique_col_values(df1):#creating the function
    for columns in df1:
        if df1[columns].dtypes=='object':# keeping only those columns whose data type is object  
            print(f"{columns}:{df1[columns].unique()}")
    


# In[22]:


print_unique_col_values(df1)# calling the function


# In[23]:


df1.replace('No internet service','No',inplace=True)


# In[24]:


print_unique_col_values(df1)#words are replaced


# In[25]:


df1.replace('No phone service','No',inplace=True)


# In[26]:


print_unique_col_values(df1)


# In[27]:


#replacing yes and no with numbers annd 1 and 0
yes_no_columns=['Partner','Dependents','PhoneService','MultipleLines',
                 'OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV',
                 'StreamingMovies','PaperlessBilling','Churn']
for col in yes_no_columns:
    df1[col].replace({'Yes' : 1,'No': 0},inplace=True)


# In[28]:


df1['gender'].replace({'Female':1,'Male':0},inplace=True)#one hot encoding


# In[29]:


df2=pd.get_dummies(data=df1,columns=['InternetService','Contract','PaymentMethod'])#dumies variable hot encoding
df2.head(5)
df2.dtypes


# # Deep learning
# 

# Scaling
# 

# In[30]:


#doing Min Max scaling to Tenure Monthlycharges and TotalCharges


# In[31]:


cols_to_scale=['tenure', 'MonthlyCharges', 'TotalCharges']
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df2[cols_to_scale]=scaler.fit_transform(df2[cols_to_scale])


# In[32]:


df2.head(5)


# In[33]:


for col in df2:
    print(f"{col}:{df2[col].unique()}")


# # Train Test Split
# 

# In[34]:


x=df2.drop('Churn',axis='columns')
y=df2['Churn']


# In[35]:


from sklearn.model_selection import train_test_split#splitting our dataset intp 80 :20 for tairning and testing respectively
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=10)


# In[36]:


len(x_train.columns)


# In[37]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Input(shape=(26,)),
    layers.Dense(20, activation='relu'),
    layers.Dense(1, activation='sigmoid')])

model.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train,epochs=5)

# need to learn about different models like Sequential
# need to learn about different activation functions
# need to learn about different optimizers
# need to learn about different lossfunctions
# need to learn about different metrices
                          


# In[38]:


model.evaluate(x_test,y_test)


# In[39]:


yp=model.predict(x_test)
yp[:5]


# In[40]:


y_test[:10]#here 2660 744 64 all are index no and 0 and 1 are prediction output


# In[41]:


y_pred=[]
for element in yp:
    if element>0.5:
        y_pred.append(1)
    else:
        y_pred.append(0) 


# In[42]:


y_pred[:10]   


# In[43]:


from sklearn.metrics import confusion_matrix, classification_report
print(classification_report(y_test,y_pred))


# In[44]:


import seaborn as sns
cm = tf.math.confusion_matrix(labels=y_test,predictions=y_pred)
plt.figure(figsize=(10,7))
ax=sns.heatmap(cm,annot=True,fmt='d')
ax.set_xlabel('Predicted')
ax.set_ylabel('Truth')
plt.show()           


# ACCURACY

# In[45]:


round((950+190)/(950+96+190+171),2)# accuracy


# PRECISION for 0 class, Precisons for the person who did not churn

# In[46]:


round((950)/(950+171),2)


# PRECISION for 1 class, Precisons for the person who did churn

# In[47]:


round((190)/(190+96),2)


# Recall for 0 class

# In[48]:


round((950)/(950+96),2)# 950+96 total no of o in churn column


# Recall for 1 class

# In[49]:


round((190)/(190+171),2)


# In[ ]:




