import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.model_selection import train_test_split  
import matplotlib.pyplot as mtp 
from scipy import stats

df = pd.read_csv("stadium.csv")
x = np.array(df["price"])
y = np.array(df["area"])


x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.25, random_state=0) 
slope,intercept,r,p,std_err=stats.linregress(x_test,y_test)

def myfunc(x):
    return slope*x+intercept

model=LinearRegression()
model.fit(x.reshape(-1,1),y)

mymodel=list(map(myfunc,x_test))

y_pred= model.predict(x_test.reshape(-1,1))  
x_pred= model.predict(x_train.reshape(-1,1))  
mtp.scatter(x_train, y_train, color="green") 
mtp.scatter(x_test,y_test,color='blue')  
mtp.plot(x_test, mymodel, color="red")    
mtp.title("price vs area")  
mtp.xlabel("area")  
mtp.ylabel("price")  
print("Correlation coefficient: ",r)
print("P_value: ",p)
print("Standard error: ",std_err)
mtp.show()