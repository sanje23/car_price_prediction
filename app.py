#%%
import numpy as np
import pandas as pd

# %%
dataset = pd.read_csv(r'C:\Users\rsanj\OneDrive\Desktop\Mlpython\datasets\car data.csv')
dataset.head()#finding first 5 datasets
# %%
dataset.info()

# %%
dataset["Transmission"].value_counts()
# %%
x=dataset.iloc[:,[1,3,4,7]].values
y=dataset.iloc[:,2].values

# %%
from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()
x[:,2]=lb.fit_transform(x[:,2])
lb1=LabelEncoder()
x[:,3]=lb1.fit_transform(x[:,3])

# %%
print(x.shape)
# %%
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
print(x_train[0,:])
# %%
from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=300,random_state=0)
regressor.fit(x_train,y_train)
# %%
accuracy=regressor.score(x_test,y_test)
print(accuracy*100,'%')
# %%
new_data=[2017,9.12,15000,"Manual"]
new_data[2]=lb.transform([new_data[2]])[0]
new_data[3]=lb1.transform([new_data[3]])[0]
# %%
print(new_data)
regressor.predict([new_data])

# %%
import pickle
pickle.dump(regressor,open('regressor.pkl','wb'))
pickle.dump(lb,open('lb','wb'))
pickle.dump(lb1,open('lb1','wb'))

