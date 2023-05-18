import pandas as pd
from sklearn.model_selection import train_test_split
import catboost as ctb

data = pd.read_csv("url.csv")
x = data.drop(columns=['Domain','Label'],axis=1)
y = data['Label']

X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.25)
catb = ctb.CatBoostClassifier()
catb_model = catb.fit(X_train,Y_train)
print("The accuracy of the model on validation set is", catb_model.score(X_test,Y_test)*100)  