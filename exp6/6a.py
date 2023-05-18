import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

df = pd.read_csv("url.csv")

X = df.iloc[:, 1:-1].values
y = df.iloc[:, -1].values

print("Total No of Columns:",X.shape[1])

X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size=0.3)

rf = RandomForestClassifier(n_estimators=3, random_state=42)
rf.fit(X_train, Y_train)

Y_pred = rf.predict(X_test)

accuracy_value = metrics.accuracy_score(Y_test, Y_pred)
print("\nAccuracy: ", accuracy_value*100)

test_data = [[0,0,1,5,0,0,0,0,0,1,1,1,0,0,1,0]]  # Example test data
predictions = []
print('\n')
for tree in rf.estimators_:
    prediction = tree.predict(test_data)
    predictions.append(prediction[0])
    print(f"Prediction of Tree {rf.estimators_.index(tree) + 1}: {'Not Malicious' if prediction[0]==0 else 'Malicious'}")

final_prediction = int(max(set(predictions), key=predictions.count))
print('\n')
if final_prediction == 0:
    print('It is not a Malicious URL')
else:
    print('It is a Malicious URL')