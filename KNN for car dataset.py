import sklearn
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import preprocessing

data = pd.read_csv("car.data")

lable = preprocessing.LabelEncoder()
lable = preprocessing.LabelEncoder()
buying = lable.fit_transform(list(data["buying"]))
maint = lable.fit_transform(list(data["maint"]))
door = lable.fit_transform(list(data["door"]))
persons = lable.fit_transform(list(data["persons"]))
lug_boot = lable.fit_transform(list(data["lug_boot"]))
safety = lable.fit_transform(list(data["safety"]))
cls = lable.fit_transform(list(data["class"]))

predict = "class"

x = list(zip(buying, maint, door, persons, lug_boot, safety))
y = list(cls)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

model = KNeighborsClassifier(n_neighbors=9)

model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
print("Accuracy :" ,acc)

predicted = model.predict(x_test)
names = ["unacc", "acc", "good", "vgood"]

for x in range(len(predicted)):
    print("Predicted: ", names[predicted[x]], "Data: ", x_test[x], "Actual: ", names[y_test[x]])

