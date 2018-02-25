# Dataset analysis of the UCI Diagnostic Breast Cancer Set
# Identifies and applies the most effective predictive model to diagnose a malignant breast tumor
# Aviel Resnick 2018

# Import neccesary libraries
import pandas
import matplotlib
import sklearn
import scipy
import numpy
import sklearn

from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt

# Machine Learning Modules from SciKit Learn
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Load dataset
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
names = ['ID', 'Diagnosis', 'M Radius', 'SE Radius', 'W Radius', 'M Texture', 'SE Texture', 'W Texture', 'M Perimeter', 'SE Perimeter', 'W Perimeter', 'M Area', 'SE Area', 'W Area', 'M Smoothness', 'SE Smoothness', 'W Smoothness', 'M Compactness', 'SE Compactness', 'W Compactness', 'M Concavity', 'SE Concavity', 'W Concavity', 'M Concave Points', 'SE Concave Points', 'W Concave Points', 'M Symmetry', 'SE Symmetry', 'W Symmetry', 'M Fractal Dimension', 'SE Fractal Dimension', 'W Fractal Dimension']
dataset = pandas.read_csv(url, names=names)
dataset = dataset.replace('M', 1)
dataset = dataset.replace('B', 0)
simplifiedDataSet = dataset.drop(dataset.columns[[0,1,3,4,6,7,9,10,12,13,15,16,18,19,21,22,24,25,27,28,30,31]], axis=1)

pandas.options.display.max_rows = 999
pandas.options.display.max_columns = 999

# Print Head (First 10 Values)
def head():
    with pandas.option_context('display.max_rows', None, 'display.max_columns', 32):
        print(simplifiedDataSet.head(10).to_string())

# Statistics about the simplified dataset
def describe():
    with pandas.option_context('display.max_rows', None, 'display.max_columns', 32):
        print(simplifiedDataSet.describe().to_string())

    # class distribution
    print(dataset.groupby('Diagnosis').size())

# box and whisker plots
def plot():
    simplifiedDataSet.plot(kind='box', subplots=True, layout=(11, 11), sharex=False, sharey=False)
    matplotlib.pyplot.show()

def main():
    #Splits the dataset 80 / 20
    array = dataset.values
    features_mean= list(simplifiedDataSet.columns[0:10])

    X = dataset.loc[:,features_mean]
    Y = dataset.loc[:, 'Diagnosis']

    X = X.astype('long')
    Y = Y.astype('long')

    validation_size = 0.20
    seed = 7
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size = validation_size, random_state = seed)

    # Test options and evaluation metric
    scoring = 'accuracy'
    models = []
    models.append(('LR', LogisticRegression()))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC()))

    # evaluate each model in turn
    results = []
    names = []
    for name, model in models:
    	kfold = model_selection.KFold(n_splits=10, random_state=seed)
    	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    	results.append(cv_results)
    	names.append(name)
    	output = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    	print(output)

    # Compare Algorithms
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    #plt.show()


    # Make predictions on validation dataset
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, Y_train)
    predictions = lda.predict(X_validation)
    print(accuracy_score(Y_validation, predictions))
    print(confusion_matrix(Y_validation, predictions))
    print(classification_report(Y_validation, predictions))

main()
