JPMML-SkLearn
=============

Java library for converting [Scikit-Learn] (http://scikit-learn.org/) models to PMML.

# Features #

* Supported Estimator and Transformer types:
  * Ensemble Methods:
    * [`ensemble.GradientBoostingClassifier`] (http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)
    * [`ensemble.GradientBoostingRegressor`] (http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html)
    * [`ensemble.RandomForestClassifier`] (http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
    * [`ensemble.RandomForestRegressor`] (http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
  * Generalized Linear Models:
    * [`linear_model.LinearRegression`] (http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
    * [`linear_model.LogisticRegression`] (http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
    * [`linear_model.Ridge`] (http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)
    * [`linear_model.RidgeClassifier`] (http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifier.html)
  * Naive Bayes:
    * [`naive_bayes.GaussianNB`] (http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html)
  * Preprocessing and Normalization:
    * [`preprocessing.Binarizer`] (http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Binarizer.html)
    * [`preprocessing.Imputer`] (http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Imputer.html)
    * [`preprocessing.LabelBinarizer`] (http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelBinarizer.html)
    * [`preprocessing.LabelEncoder`] (http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html)
    * [`preprocessing.MinMaxScaler`] (http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html)
    * [`preprocessing.OneHotEncoder`] (http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)
    * [`preprocessing.StandardScaler`] (http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
  * Decision Trees:
    * [`tree.DecisionTreeClassifier`] (http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
    * [`tree.DecisionTreeRegressor`] (http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)
* Production quality:
  * Complete test coverage.
  * Fully compliant with the [JPMML-Evaluator] (https://github.com/jpmml/jpmml-evaluator) library.

# Prerequisites #

### The Python side of operations

* Python 3.4 or newer.
* [`scikit-learn`] (https://pypi.python.org/pypi/scikit-learn) 0.16.0 or newer.
* [`pandas`] (https://pypi.python.org/pypi/pandas) 0.16.2 or newer.
* [`sklearn-pandas`] (https://pypi.python.org/pypi/sklearn-pandas) 0.0.10 or newer.
* [`joblib`] (https://pypi.python.org/pypi/joblib) 0.8.4 or newer.
* [`numpy`] (https://pypi.python.org/pypi/numpy) 1.9.2 or newer.

Python installation can be validated as follows:

```python
import sklearn, pandas, sklearn_pandas, joblib, numpy

print(sklearn.__version__)
print(pandas.__version__)
print(sklearn_pandas.__version__)
print(joblib.__version__)
print(numpy.__version__)
```

### The JPMML-SkLearn side of operations

* Java 1.7 or newer.

# Installation #

Enter the project root directory and build using [Apache Maven] (http://maven.apache.org/):
```
mvn clean install
```

The build produces an executable uber-JAR file `target/converter-executable-1.0-SNAPSHOT.jar`.

# Usage #

A typical workflow can be summarized as follows:

1. Use Python to train a model.
2. Serialize the model in `pickle` data format to a file in a local filesystem.
3. Use the JPMML-SkLearn command-line converter application to turn the pickle file to a PMML file.

### The Python side of operations

Load data to a `pandas.DataFrame` object:
```python
import pandas

iris_df = pandas.read_csv("Iris.csv")
```

Describe data and data pre-processing actions by creating an appropriate `sklearn_pandas.DataFrameMapper` object:
```python
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper

iris_mapper = DataFrameMapper([
    ("Sepal.Length", StandardScaler()),
    ("Sepal.Width", StandardScaler()),
    ("Petal.Length", StandardScaler()),
    ("Petal.Width", StandardScaler()),
    ("Species", None)
])

iris_df = iris_mapper.fit_transform(iris_df)
```

Train an appropriate estimator object:
```python
from sklearn.ensemble.forest import RandomForestClassifier

iris_X = iris_df[:, 0:4]
iris_y = iris_df[:, 4]

iris_forest = RandomForestClassifier(min_samples_leaf = 5)
iris_forest.fit(iris_X, iris_y)
```

Serialize the `sklearn_pandas.DataFrameMapper` object and estimator object in `pickle` data format:
```python
from sklearn.externals import joblib

joblib.dump(iris_mapper, "mapper.pkl")
joblib.dump(iris_forest, "estimator.pkl")
```

Please see the test script file [main.py] (https://github.com/jpmml/jpmml-sklearn/blob/master/src/test/resources/main.py) for more classification (binary and multi-class) and regression workflows.

### The JPMML-SkLearn side of operations

Converting the estimator pickle file `estimator.pkl` to a PMML file `estimator.pmml`:
```
java -jar target/converter-executable-1.0-SNAPSHOT.jar --pkl-input estimator.pkl --pmml-output estimator.pmml
```

Converting the `sklearn_pandas.DataFrameMapper` pickle file `mapper.pkl` and the estimator pickle file `estimator.pkl` to a PMML file `mapper-estimator.pmml`:
```
java -jar target/converter-executable-1.0-SNAPSHOT.jar --pkl-mapper-input mapper.pkl --pkl-estimator-input estimator.pkl --pmml-output mapper-estimator.pmml
```

Getting help:
```
java -jar target/converter-executable-1.0-SNAPSHOT.jar --help
```

# License #

JPMML-SkLearn is dual-licensed under the [GNU Affero General Public License (AGPL) version 3.0] (http://www.gnu.org/licenses/agpl-3.0.html) and a commercial license.

# Additional information #

Please contact [info@openscoring.io] (mailto:info@openscoring.io)