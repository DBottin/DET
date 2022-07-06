from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import pandas as pd

#Daten einlesen
dataset = pd.read_csv("https://raw.githubusercontent.com/DBottin/DET/main/DATA.csv")

#Datensätze zu Integer machen, damit DecisionTreeClassifier funktioniert

dataset["age"] = dataset["age"].astype('category')
dataset["age"] = dataset.age.cat.codes
#<=30 wird zu 0, 31...40 wird zu 1 und >40 wird zu 2

dataset["income"] = dataset["income"].astype('category')
dataset["income"] = dataset.income.cat.codes
#Low wird zu 0, Medium wird zu 1 und High wird zu 2

dataset["student"] = dataset["student"].astype('category')
dataset["student"] = dataset.student.cat.codes
#No wird zu 0, Yes, wird zu 1

dataset["credit_rating"] = dataset["credit_rating"].astype('category')
dataset["credit_rating"] = dataset.credit_rating.cat.codes
#Fair wird zu 0, Excellent wird zu 1

X = dataset.drop(['buys_computer'], axis=1)
y = dataset["buys_computer"]

tree_clf = DecisionTreeClassifier(criterion="gini")
tree_clf.fit(X, y)

export_graphviz(
         tree_clf,
         out_file="C:/Users\Rayma\Desktop\gini.dot",
         feature_names=list(X.columns),
         class_names=["no", "yes"],
         rounded=True,
         filled=True
 )

