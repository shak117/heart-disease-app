#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # About the Dataset:
# ### Features
# 1.age
# 
# 2.sex
# 
# 3.chest pain type (4 values)
# 
# 4.resting blood pressure
# 
# 5.serum cholestoral in mg/dl
# 
# 6.fasting blood sugar > 120 mg/dl
# 
# 7.resting electrocardiographic results (values 0,1,2)
# 
# 8.maximum heart rate achieved
# 
# 9.exercise induced angina
# 
# 10.oldpeak = ST depression induced by exercise relative to rest
# 
# 11.the slope of the peak exercise ST segment
# 
# 12.number of major vessels (0-3) colored by flourosopy
# 
# 13.thal: 0 = normal; 1 = fixed defect; 2 = reversable defect The names and social security numbers of the patients were recently removed from the database, replaced with dummy values.

# # Importing Necessary Libraries:

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)


# # Loading Data:

# In[3]:


df=pd.read_csv(r"C:\Users\Admin1\OneDrive\Desktop\heart.csv")


# In[4]:


#df.head()


# # Data Preprocessing:

# In[5]:


#df.shape


# In[6]:


#df.info()


# In[7]:


df.isna().sum()


# # Exploratory Data Analysis:

# In[8]:


#df.describe().T


# In[9]:


#df.hist(bins=20,figsize=(16,16))
plt.show()


# In[10]:


#plt.figure(figsize=(12,8))
#sns.heatmap(df.corr(),annot=True)
#plt.title("Correlation Heatmap")
#plt.show()


# In[11]:


#sns.countplot(data=df,x=df['target'])
#plt.title("Target Variable Distribution")
#plt.show()


# In[12]:


fig,axes=plt.subplots(7,2,figsize=(20,12))
axes=axes.flatten()
for i,col in enumerate(df.columns):
    sns.boxplot(x=df[col],ax=axes[i])
    axes[i].set_ylabel(f"{col}")
plt.show()


# # Splitting The Data:

# In[13]:


x=df.drop(columns=['target'],axis=1)
y=df['target']


# In[14]:


scaler=StandardScaler()
x_scaled=scaler.fit_transform(x)


# In[15]:


x_train,x_test,y_train,y_test=train_test_split(x_scaled,y,test_size=0.2,random_state=42)


# # Trainig Multiple Models:

# In[16]:


lr=LogisticRegression()


# In[17]:


lr.fit(x_train,y_train)


# In[18]:


lr_pred=lr.predict(x_test)


# In[19]:


print("Logistic Regression Accuracy",accuracy_score(y_test,lr_pred))


# In[20]:


print("Classification Report(Logistic Regression):")
print(classification_report(y_test,lr_pred))


# In[21]:


knn=KNeighborsClassifier()


# In[22]:


knn.fit(x_train,y_train)


# In[23]:


knn_pred=knn.predict(x_test)


# In[24]:


print("KNN Accuracy:",accuracy_score(y_test,knn_pred))


# In[25]:


print("Classification Report(KNN):")
print(classification_report(y_test,knn_pred))


# # Trying Ensemble Models:

# In[26]:


models={
    'Random Forest':RandomForestClassifier(),
    'Gradient Boosting':GradientBoostingClassifier(),
    'XGBoost':XGBClassifier()
}


# In[27]:


for name,model in models.items():
    model.fit(x_train,y_train)
    y_pred=model.predict(x_test)
    acc=accuracy_score(y_test,y_pred)
    cl_rep=classification_report(y_test,y_pred)
    print(f"{name}:{acc}")
    print('Classification Report:')
    print(f"{cl_rep}")


# # Hyperparameter Tuning:

# In[28]:


params={
    'n_estimators':[100,200,300,500,1000],
    'learning_rate':[0.01,0.1],
    'max_depth':[3,5,7], 
    'subsample':[0.5,0.8,1],
    'colsample_bytree':[0.5,0.8,1]
}


# In[29]:


grid=GridSearchCV(XGBClassifier(objective='binary:logistic',eval_metric='logloss'),params,scoring='accuracy',cv=5,n_jobs=-1)


# In[30]:


grid.fit(x_train,y_train)


# In[31]:


grid.best_params_


# In[32]:


#grid.best_score_


# In[33]:


best_xgb=grid.best_estimator_


# In[34]:


best_xgb.fit(x_train,y_train)


# In[35]:


best_xgb_pred=best_xgb.predict(x_test)


# In[36]:


print("Accuracy:",accuracy_score(y_test,best_xgb_pred))


# In[37]:


print("Classification Report")
print(classification_report(y_test,best_xgb_pred))


# In[38]:


cm=confusion_matrix(y_test,best_xgb_pred)


# In[39]:


sns.heatmap(cm,annot=True,fmt='d')
plt.title("Actual")
plt.ylabel("Predicted")
plt.show()


# ### So The best accuracy we wre getting is 98.53%.

# In[40]:


df


# In[41]:


input_data=df.copy()


# In[42]:


input_data =(1020,	59,	1,	1,	140,	221,	0,	1,	164,	1,	0.0,	2,	0 )
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
prediction = model.predict(input_data_reshaped)
print(prediction)
if (prediction[0]==0):
  print('The Person does not have heart disease')
else:
  print('The Person has heart disease')


# In[43]:


 


# In[44]:


from fastapi import FastAPI
import numpy as np
import pickle

app = FastAPI(title="Heart Disease Prediction API")
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # allow frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# load trained model
model = pickle.load(open("heart_model.pkl", "rb"))  # change filename if needed


@app.post("/predict")
def predict_heart_disease(data: dict):
    """
    Expected input keys:
    age, sex, cp, trestbps, chol, fbs, restecg,
    thalach, exang, oldpeak, slope, ca, thal
    """

    input_data = (
        data["age"],
        data["sex"],
        data["cp"],
        data["trestbps"],
        data["chol"],
        data["fbs"],
        data["restecg"],
        data["thalach"],
        data["exang"],
        data["oldpeak"],
        data["slope"],
        data["ca"],
        data["thal"]
    )

    input_array = np.asarray(input_data).reshape(1, -1)
    prediction = model.predict(input_array)

    if prediction[0] == 0:
        return {"result": "The Person does NOT have heart disease"}
    else:
        return {"result": "The Person HAS heart disease"}


# In[45]:


#import pickle
#odel = pickle.load(open("heart_model.pkl", "rb"))
#int(type(model))


# In[46]:


#heart_model.pkl=model.copy()


# In[47]:


#model.save_model("heart_model.json")


# In[48]:


import pickle

with open("heart_model.pkl", "wb") as file:
    pickle.dump(model, file)


# In[49]:


import os
print(os.listdir())


# In[50]:


model = pickle.load(open("heart_model.pkl", "rb"))
print(model)


# In[51]:


import pickle
from fastapi import FastAPI
import numpy as np

app = FastAPI()

model = pickle.load(open("heart_model.pkl", "rb"))

@app.post("/predict")
def predict(data: dict):
    features = np.array([
        data["age"], data["sex"], data["cp"], data["trestbps"],
        data["chol"], data["fbs"], data["restecg"], data["thalach"],
        data["exang"], data["oldpeak"], data["slope"], data["ca"], data["thal"]
    ]).reshape(1, -1)

    pred = model.predict(features)
    return {"prediction": int(pred[0])}


# In[52]:


import pickle
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load model safely
with open("heart_model.pkl", "rb") as f:
    model = pickle.load(f)

# Input schema
class HeartInput(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

@app.post("/predict")
def predict(data: HeartInput):
    features = np.array([[
        data.age, data.sex, data.cp, data.trestbps,
        data.chol, data.fbs, data.restecg, data.thalach,
        data.exang, data.oldpeak, data.slope, data.ca, data.thal
    ]])

    prediction = model.predict(features)

    return {
        "prediction": int(prediction[0])
    }
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

df = pd.read_csv("heart.csv")

X = df.drop("target", axis=1)
y = df["target"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = XGBClassifier()
model.fit(X_scaled, y)

# SAVE EVERYTHING TO PKL
with open("heart_model.pkl", "wb") as f:
    pickle.dump({
        "model": model,
        "scaler": scaler
    }, f)

print("Model saved successfully ✅")


# In[ ]:heart_model.pkl





# In[ ]:




