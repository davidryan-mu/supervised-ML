# Import loading spinner
import time
from yaspin import yaspin
from yaspin.spinners import Spinners

# Importing the modelling libraries
from sklearn.neural_network import MLPClassifier
import joblib
import pandas as pd

# Import when using validation set from training data
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix, plot_roc_curve
# import matplotlib.pyplot as plt

joblib_file = "joblib_model.pkl"

def make_all():
    with yaspin(text = 'Running script and training model...' ) as sp:
        # Importing the training and test datasets
        train_io = pd.read_csv('./train.txt', sep=' ', header=None)
        test_in = pd.read_csv('./test.txt', sep=' ', header=None)
        sp.write('> Data read from files')

        # Splitting the training sets inputs and outputs
        X = train_io.iloc[:, 0:12].values
        y = train_io.iloc[:, 12].values
        sp.write('> Input/output split from training set')

        # Split the training set when using validation set.
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 4)
        # sp.write('> Split into 70% and 30% test data')

        # Grab input values of test set
        test_values = test_in.iloc[:, :].values

        # Building and training the model
        sp.write('> Building classification model...')
        classifier = MLPClassifier(hidden_layer_sizes=(150,100,50), max_iter=300,activation = 'relu',solver='adam',random_state=1)
        sp.write('  ~ DONE: Built classification model')

        sp.write('> Training model - this could take a while...')
        start_time = time.time()
        classifier.fit(X, y)
        elapsed_time = time.time() - start_time
        sp.write('  ~ DONE: Trained model')
        sp.write('     -- Elapsed time: ' + str(elapsed_time))

        # Save trained model to file
        joblib.dump(classifier, joblib_file)

        # Predicting the Test set results
        sp.write('> Predicting output...')
        y_pred = classifier.predict(test_values)
        sp.write('  ~ DONE: Predicted output')

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

    print()
    print('New file created/overwritten at test-out.txt \n Number of positives (1\'s): ' +
        str(oneCount) + '\n Number of negatives (0\'s): ' + str(zeroCount))

def make_test_out():
    with yaspin(text = 'Running script without retraining...' ) as sp:
        # Load model from file
        classifier = joblib.load(joblib_file)
        sp.write('> Read trained model from files')

        # Importing the test dataset
        test_in = pd.read_csv('./test.txt', sep=' ', header=None)
        sp.write('> Data read from files')

        # Grab input values of test set
        test_values = test_in.iloc[:, :].values

        # Predicting the Test set results
        sp.write('> Predicting output...')
        y_pred = classifier.predict(test_values)
        sp.write('  ~ DONE: Predicted output')

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

    print()
    print('New file created/overwritten at test-out.txt \n Number of positives (1\'s): ' +
        str(oneCount) + '\n Number of negatives (0\'s): ' + str(zeroCount))

# Use to find metrics on a validation set      
# print()
# print('Confusion matrix: ')
# print()
# print(confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred))
# print('Accuracy:', accuracy_score(y_test, y_pred))
# print('ROC Score:', roc_auc_score(y_test, y_pred))
# plot_roc_curve(classifier, X_test, y_test)
# plt.show()