# BONE - Supervised ML Assignment

A script that takes in a training set, trains a Multi-Layer Perceptron and tests itself on the given test set. Test output is sent to `test-out.txt` which is overwritten if it exists, and created if it does not. This is formatted with one prediction per line.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install dependencies. **NOTE:** Due to scikit-learn [not yet supporting Python 3.9](https://github.com/scikit-learn/scikit-learn/issues/18621#issuecomment-708957810), it is recommended to run this script using an earlier version of Python. (e.g. [Python 3.8.6](https://www.python.org/downloads/release/python-386/))

```bash
pip install -r requirements.txt
```

This command will install the necessary dependencies which are as follows:
````
pandas==1.1.5
scikit_learn==0.23.2
yaspin==1.2.0
````
* **Pandas** for reading in datasets
* **Scikit-learn** for building and training model
* **Yaspin** for real time progress spinner (training model can take some time)


## Usage
**To regenerate `test-out.txt` WITHOUT retraining**
1. copy `train.txt` and `test.txt` into the directory
2. Navigate to directory in command prompt
3. Type in the following commands to open an interactive python session and run appropriate script
 ````
python
import script
script.make_test_out()
````

**To retrain classifier and regenerate `test-out.txt`**
(Takes a while...)
1. copy `train.txt` and `test.txt` into the directory
2. Navigate to directory in command prompt
3. Type in the following commands to open an interactive python session and run appropriate script
 ````
python
import script
script.make_all()
````



## License
[MIT](https://choosealicense.com/licenses/mit/)