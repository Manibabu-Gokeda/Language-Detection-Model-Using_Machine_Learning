#!/usr/bin/env python
# coding: utf-8

# # Step 1: Collection of Data and Loading the data

# In[1]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
data=pd.read_csv("Language Detection.csv")
print(data.head())


# #  Step 2: Data Processing 

# In[2]:


data.shape # It Describes the rows and columns of the dataset


# In[3]:


data.isnull().sum() # Counts the null values in the columns of the dataset


# In[4]:


data["Language"].value_counts() #It shows individual values in the column


# In[5]:


data.select_dtypes(include="object").columns


# # Step 3: Data Visualisation

# In[6]:


language = data["Language"].value_counts().reset_index()

plt.figure(figsize=(10,10))
labels= language['index']

plt.pie(language["Language"], labels= labels, autopct='%.1f%%', textprops={'fontsize': 15})
plt.show()


# # Step 4: Seperating Training Set and Testing Set

# In[7]:


x = np.array(data["Text"])
y = np.array(data["Language"])
cv = CountVectorizer()
X = cv.fit_transform(x)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.33, 
                                                    random_state=42)


# # Step 5: Model Implementation 

# In[8]:


model = MultinomialNB()
model.fit(X_train,y_train)
model.score(X_test,y_test)


# # Step 6: Model Evalution

# In[ ]:


user = input("Enter a Text: ")
if(user.isdigit()):
    print("Enter valid language")

else:
    data = cv.transform([user]).toarray()
    output = model.predict(data)
    print(output)


# In[ ]:





# In[ ]:




