# Import loading spinner
import time
from yaspin import yaspin
from yaspin.spinners import Spinners

# Import time estimator
from scitime import Estimator

# Importing the modelling libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
import numpy as np
import pandas as pd


with yaspin(text = 'Running script...' ) as sp:
    estimator = Estimator()

    # Importing the training and test datasets
    train_io = pd.read_csv('./train-io.txt', sep=' ', header=None)
    test_in = pd.read_csv('./test-in.txt', sep=' ', header=None)
    sp.write('> Data read from files')

    # Splitting the training sets inputs and outputs
    X = train_io.iloc[0:100000, 0:12].values
    y = train_io.iloc[0:100000, 12].values
    sp.write('> Input/output split')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
    sp.write('> Split into 80% and 20% test data')

    # Grab input values of test set
    # test_values = test_in.iloc[:, :].values

    # Building and training the model
    sp.write('> Building sigmoid SVC...')
    classifier = SVC(gamma = 'auto', kernel = 'sigmoid')
    sp.write('  ~ DONE: Built sigmoid SVC')

    sp.write('> Training model...')
    start_time = time.time()
    classifier.fit(X_train, y_train)
    elapsed_time = time.time() - start_time
    sp.write('  ~ DONE: Trained model')
    sp.write('     -- Elapsed time: ' + str(elapsed_time))

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)

    # Print test output to new file and count occurences
    zeroCount = 0
    oneCount = 0
    newFile = open('test-out.txt', 'w')
    for element in y_pred:
        if (element == 0):
            zeroCount += 1
        elif (element == 1):
            oneCount += 1

        newFile.write(str(element))
        newFile.write('\n')
    newFile.close()

    sp.text = 'Successfully trained model and predicted values on test set.'
    sp.ok('✔')

print('New file created/overwritten at test-out.txt \n Number of positives (1\'s): ' +
      str(oneCount) + '\n Number of negatives (0\'s): ' + str(zeroCount))

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
