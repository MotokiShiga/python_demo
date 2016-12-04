import pandas as pd
from sklearn import datasets
from sklearn import tree

# load dataset IRIS
iris = datasets.load_iris()

# training
classifier = tree.DecisionTreeClassifier()
classifier = classifier.fit(iris.data, iris.target)

# output decision tree
with open('dt_graph.dot', 'w') as file_out:
    file_out = tree.export_graphviz(classifier, out_file=file_out)

#### To output into a pdf file, run a Graphviz command:
# > dot -Tpdf dt_graph.dot -o dt_graph.pdf 
