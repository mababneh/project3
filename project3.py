# mohammed Ababneh
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time
def main():
    clock = True
    DatasetNM = input(
        "Please Enter the name of DataSet (DigitData,REALDISP,otherData)\n")
    if DatasetNM == "Digit":
        digits = load_digits()   #load data already in Sklearn
        T = digits.data          #load the values of feature
        P = digits.target        #load the values of dependent variable
        scaler = StandardScaler().fit(T)
        rescaledX = scaler.transform(T)

        print("---------------------------------------------------------------------------------------------------")
        X, Xtest, y, ytest = train_test_split(T, P, random_state=1, test_size=.3)
    if DatasetNM == "REALDISP":
        # this is REALDISP Activity Recognition Dataset Data Set
        train = pd.read_csv('/Users/Mohammed/Desktop/REALDISP.csv', header=None)
        trainraws = []
        i = 0
        # the number of the input
        for i in range(119):
            trainraws.append(i)
        print(trainraws)
        # dataset feature values and target value
        T = train.iloc[0:179036, trainraws].values
        P = train.iloc[0:179036, 119].values
        scaler = StandardScaler().fit(T) # standardization dataset
        rescaledX = scaler.transform(T)
        #divide the dataset into train data and test data
        X, Xtest, y, ytest = train_test_split(T, P, random_state=1, test_size=.3)
    if DatasetNM=="otherData":
        # it is used to enter other dataset by usinf the path of dataset file and number of coloumn
        trainraws = []
        DatasetPath = input("Please Enter the path of DataSet\n")
        train = pd.read_csv(DatasetPath, header=None)
        numberofC = input("Please Enter number of coulmns in dataset\n")
        features=int(numberofC)-1
        for i in range(features):
            trainraws.append(i)
        print(trainraws)
        T = train.iloc[:, trainraws].values
        P = train.iloc[:, features].values
        scaler = StandardScaler().fit(T)
        rescaledX = scaler.transform(T)

        X, Xtest, y, ytest = train_test_split(rescaledX, P, random_state=1, test_size=.3)




    def CheckingClassifer(ClassiferName):
        if ClassiferName=="perceptron":
            #perceptron classifer with fit and predict function
            #computing Running Time and Accuracy
            print("-----------------------------Perceptron------------------------------------------------")
            start_time = time.time()
            pop = Perceptron(penalty=None, alpha=0.0001, fit_intercept=True, max_iter=50, tol=None, shuffle=True,
                             verbose=0, eta0=0.01, n_jobs=None, random_state=0, early_stopping=False,
                             validation_fraction=0.1,
                             n_iter_no_change=5, class_weight=None,
                             warm_start=False)
            pop.fit(X, y)
            pop.predict(Xtest)
            print("--- %s seconds ---" % (time.time() - start_time))

            print(pop.score(Xtest, ytest))
        elif ClassiferName=="RBFSVC":
            #SVM classifier with RBF Kernel
            # computing Running Time and Accuracy
            print("------------------------NON linear SVC-------------------------------------")
            start_time = time.time()
            pip = SVC(gamma='auto', C=15)
            df = pip.fit(X, y)
            dd = pip.predict(Xtest)
            print(dd)

            print("--- %s seconds ---" % (time.time() - start_time))

            print(pip.score(Xtest, ytest))
        elif ClassiferName=="LinerSVC":
            #the SVC Classifier with linear Kernel with fit and predict function
            # computing Running Time and Accuracy
            print("------------------------linear SVC-------------------------------------")
            start_time = time.time()
            pip = SVC(gamma='auto', kernel='linear')
            df = pip.fit(X, y)
            dd = pip.predict(Xtest)
            print(dd)

            print("--- %s seconds ---" % (time.time() - start_time))

            print(pip.score(Xtest, ytest))
        elif ClassiferName=="TreeDescion":
            #the descision tree classifer with fit and predict function
            # computing Running Time and Accuracy
            print("-----------------------------TreeDescion---------------------------------------")
            start_time = time.time()
            clf = DecisionTreeClassifier(random_state=0, max_depth=15)
            df = clf.fit(X, y)
            dd = clf.predict(Xtest)
            print(dd)

            print("--- %s seconds ---" % (time.time() - start_time))

            print(clf.score(Xtest, ytest))
        elif ClassiferName=="KNN":
            #the K-neaset neighbors with fit and predict function
            # computing Running Time and Accuracy
            print("--------------------------KNN--------------------------------------------")
            start_time = time.time()
            model = KNeighborsClassifier(n_neighbors=1)
            nr = model.fit(X, y)
            nrd = model.predict(Xtest)
            print("--- %s seconds ---" % (time.time() - start_time))

            print("Accuracy:", metrics.accuracy_score(ytest, nrd))
        elif ClassiferName=="LG":
            #this is logestic Regression Classifier with fit and predict method
            # computing Running Time and Accuracy
            print("-------------------------LG---------------------------------------------")
            start_time = time.time()
            model2 = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True,
                                        intercept_scaling=1, class_weight=None, random_state=None, max_iter=500,
                                        solver='liblinear')
            nr1 = model2.fit(X, y)
            nrd1 = model2.predict(Xtest)

            print("--- %s seconds ---" % (time.time() - start_time))

            print("Accuracy:", metrics.accuracy_score(ytest, nrd1))

    def ERRoRCheckingClassifer():
        # this function is used to check the name of classifier is correct or not
        # if the user input 0 mean the program will be terninated
        while clock==True:
            ClassiferName = input("please Enter The ClassiferName {perceptron,LG,KNN,TreeDescion,RBFSVC,LinerSVC}:\n")
            if ((ClassiferName=="perceptron") or (ClassiferName=="LG") or (ClassiferName=="KNN") or(ClassiferName=="TreeDescion") or(ClassiferName=="RBFSVC") or(ClassiferName=="LinerSVC")):
                CheckingClassifer(ClassiferName)
            elif ClassiferName == str(0):
                print("Thank You for Using My program")
                break
            else:
                print("please Enter the name of the file correctly\n")

    ERRoRCheckingClassifer()


if __name__ == "__main__":
    main()