import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv('PlayTennis.csv')
x = df.drop('Play', axis=1)
y = df['Play']

le = LabelEncoder()
x = x.apply(le.fit_transform)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier(criterion='entropy', splitter='best')

clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
print("Accuracy is ", accuracy_score(y_test,y_pred)*100)

plt.figure(figsize=(20,10))
plot_tree(clf, filled=True, feature_names=x.columns, class_names=['0', '1'])
plt.show()