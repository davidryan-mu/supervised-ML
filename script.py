# Importing the libraries
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd

# Importing the training and test datasets
train_io = pd.read_csv('./train-io.txt', sep=' ', header=None)
test_in = pd.read_csv('./test-in.txt', sep=' ', header=None)

# Splitting the training sets inputs and outputs
train_in = train_io.iloc[:, 0:12].values
train_out = train_io.iloc[:, 12].values

# Grab input values of test set
test_values = test_in.iloc[:, :].values

# Building and training the model
classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(train_in, train_out)

# Predicting the Test set results
test_out = classifier.predict(test_values)

# Print test output to new file and count occurences
zeroCount = 0
oneCount = 0
newFile = open('test-out.txt', 'w')
for element in test_out:
    if (element == 0):
        zeroCount += 1
    elif (element == 1):
        oneCount += 1

    newFile.write(str(element))
    newFile.write('\n')
newFile.close()

print('New file created/overwritten at test-out.txt \n Number of positives (1\'s): ' +
      str(oneCount) + '\n Number of negatives (0\'s): ' + str(zeroCount))
