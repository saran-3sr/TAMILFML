import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.naive_bayes import GaussianNB

data = pd.read_csv('Average.csv')
data = pd.get_dummies(data, columns=['Team1', 'Team2', 'Venue'])
print(data)

df = pd.read_csv('Average.csv')
X = pd.get_dummies(df.drop('Team1_Status', axis=1))
print(X)
y = df['Team1_Status']
X_train, X_test, y_train, y_test = train_test_split(data.drop('Team1_Status', axis=1), data['Team1_Status'], test_size=0.2, random_state=42)
model = GaussianNB()
model.fit(X, y)
team1 = input("Enter Team 1: ")
team2 = input("Enter Team 2: ")
venue = input("Enter Venue: ")

X_new = pd.DataFrame({'Team1_' + team1: [1], 'Team2_' + team2: [1], 'Venue_' + venue: [1]})
X_new = X_new.reindex(columns=X.columns, fill_value=0)
y_new = model.predict(X_new)
y_pred = model.predict(X_test)
print(f"The predicted Team1_Status is {y_new[0]}")
accuracy = accuracy_score(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print('Accuracy:', accuracy)
print('Confusion Matrix:\n', confusion_mat)
print('Classification Report:\n', class_report)