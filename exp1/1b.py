import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df= pd.DataFrame({"Match_Number" : range(1,11),
                     "No_of_sixes"  : np.random.randint(5, 20, size=10),
                     "No_of_fours"  : np.random.randint(5, 30, size=10)
                     })
print(df)
df.to_csv('sport.csv')
plt.plot(df['Match_Number'],df[ "No_of_sixes"]) 
plt.bar(df['Match_Number'],df[ "No_of_fours"], color ='maroon',width = 0.4)
plt.scatter(df['Match_Number'],df[ "No_of_sixes"]) 
plt.xlabel('Match_Number')
plt.ylabel('No_of_sixes')
plt.pie(df["No_of_fours"],labels=df['Match_Number'])
plt.legend(title = "No of sixes in each match")