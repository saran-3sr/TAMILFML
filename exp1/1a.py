import numpy as np
import pandas as pd
arr=np.array([['mango',2,50],['orange',5,60],['apple',10,70]])
df=pd.DataFrame(arr,columns=['Fruit Name','Kg','Cost per Kg'])
print(df)
df['Total Cost']=df['Kg'].astype(int)*df['Cost per Kg'].astype(int)
print(df)
df.to_csv('fruits.csv')