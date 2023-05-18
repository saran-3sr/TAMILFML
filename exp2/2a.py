import matplotlib.pyplot as plt
import math
import pandas as p
df = p.read_csv("stadium.csv")
x = df["price"].tolist()[:160]
y = df["area"].tolist()[:160]
sum_x = sum(x)
sum_y = sum(y)
xy = []
def square_list(x):
    return x*x
for a in range(len(x)):
    pro = x[a]*y[a]
    xy.append(pro)
sum_xy = sum(xy)
sum_x_squared = sum(list(map(square_list,x)))
a = (len(x) * sum_xy - sum_x * sum_y) / (len(x) * sum_x_squared - sum_x ** 2)
b = (sum_y - a * sum_x) / len(x)
y_pred = [a*x[i] + b for i in range(len(x))]
x_test = df["price"].tolist()[160:200]
y_test = df["area"].tolist()[160:200]
y_test_pred = [a * x_test[i] + b for i in range(len(x_test))] 
numerator=sum((y_test[i] - y_test_pred[i])**2 for i in range(len(x_test)))
denominator=sum((y_test[i] - sum(y_test) / len(x_test))**2 for i in range(len(x_test)))
accuracy = numerator / denominator
rmse = math.sqrt(sum((y_test[i] - y_test_pred[i]) ** 2 for i in range(len(x_test))) / len(x_test))
print("Accuracy: ", accuracy)
print("RMSE: ",rmse)
plt.scatter(x, y, color='red')
plt.scatter(x_test, y_test, color='green')
plt.plot(x, y_pred, color='black', linewidth=2)
plt.show()