JPMML-SkLearn
=============

Java library and command-line application for converting [Scikit-Learn](http://scikit-learn.org/) models to PMML.

# Features #

* Supported Estimator and Transformer types:
  * Clustering:
    * [`cluster.KMeans`](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
    * [`cluster.MiniBatchKMeans`](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html)
  * Matrix Decomposition:
    * [`decomposition.PCA`](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
    * [`decomposition.IncrementalPCA`](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.IncrementalPCA.html)
  * Discriminant Analysis:
    * [`discriminant_analysis.LinearDiscriminantAnalysis`](http://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html)
  * Dummies:
    * [`dummy.DummyClassifier`](http://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html)
    * [`dummy.DummyRegressor`](http://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyRegressor.html)
  * Ensemble Methods:
    * [`ensemble.AdaBoostRegressor`](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html)
    * [`ensemble.BaggingClassifier`](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html)
    * [`ensemble.BaggingRegressor`](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingRegressor.html)
    * [`ensemble.ExtraTreesClassifier`](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html)
    * [`ensemble.ExtraTreesRegressor`](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html)
    * [`ensemble.GradientBoostingClassifier`](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)
    * [`ensemble.GradientBoostingRegressor`](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html)
    * [`ensemble.IsolationForest`](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)
    * [`ensemble.RandomForestClassifier`](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
    * [`ensemble.RandomForestRegressor`](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
    * [`ensemble.VotingClassifier`](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html)
  * Feature Extraction:
    * [`feature_extraction.DictVectorizer`](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.DictVectorizer.html)
    * [`feature_extraction.text.CountVectorizer`](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)
    * [`feature_extraction.text.TfidfVectorizer`](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
  * Feature Selection:
    * [`feature_selection.GenericUnivariateSelect`](http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.GenericUnivariateSelect.html) (only via `sklearn2pmml.SelectorProxy`)
    * [`feature_selection.RFE`](http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html) (only via `sklearn2pmml.SelectorProxy`)
    * [`feature_selection.RFECV`](http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html) (only via `sklearn2pmml.SelectorProxy`)
    * [`feature_selection.SelectFdr`](http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFdr.html) (only via `sklearn2pmml.SelectorProxy`)
    * [`feature_selection.SelectFpr`](http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFpr.html) (only via `sklearn2pmml.SelectorProxy`)
    * [`feature_selection.SelectFromModel`](http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html) (either directly or via `sklearn2pmml.SelectorProxy`)
    * [`feature_selection.SelectFwe`](http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFwe.html) (only via `sklearn2pmml.SelectorProxy`)
    * [`feature_selection.SelectKBest`](http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html) (either directly or via `sklearn2pmml.SelectorProxy`)
    * [`feature_selection.SelectPercentile`](http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectPercentile.html) (only via `sklearn2pmml.SelectorProxy`)
    * [`feature_selection.VarianceThreshold`](http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.VarianceThreshold.html) (only via `sklearn2pmml.SelectorProxy`)
  * Generalized Linear Models:
    * [`linear_model.ElasticNet`](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html)
    * [`linear_model.ElasticNetCV`](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNetCV.html)
    * [`linear_model.Lasso`](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html)
    * [`linear_model.LassoCV`](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html)
    * [`linear_model.LinearRegression`](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
    * [`linear_model.LogisticRegression`](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
    * [`linear_model.LogisticRegressionCV`](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html)
    * [`linear_model.Ridge`](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)
    * [`linear_model.RidgeCV`](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html)
    * [`linear_model.RidgeClassifier`](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifier.html)
    * [`linear_model.RidgeClassifierCV`](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifierCV.html)
    * [`linear_model.SGDClassifier`](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html)
    * [`linear_model.SGDRegressor`](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html)
  * Naive Bayes:
    * [`naive_bayes.GaussianNB`](http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html)
  * Nearest Neighbors:
    * [`neighbors.KNeighborsClassifier`](http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
    * [`neighbors.KNeighborsRegressor`](http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html)
  * Pipelines:
    * [`pipeline.FeatureUnion`](http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.FeatureUnion.html)
    * [`pipeline.Pipeline`](http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)
  * Neural network models:
    * [`neural_network.MLPClassifier`](http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)
    * [`neural_network.MLPRegressor`](http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html)
  * Preprocessing and Normalization:
    * [`preprocessing.Binarizer`](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Binarizer.html)
    * [`preprocessing.FunctionTransformer`](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.FunctionTransformer.html)
    * [`preprocessing.Imputer`](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Imputer.html)
    * [`preprocessing.LabelBinarizer`](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelBinarizer.html)
    * [`preprocessing.LabelEncoder`](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html)
    * [`preprocessing.MaxAbsScaler`](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MaxAbsScaler.html)
    * [`preprocessing.MinMaxScaler`](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html)
    * [`preprocessing.OneHotEncoder`](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)
    * [`preprocessing.PolynomialFeatures`](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html)
    * [`preprocessing.RobustScaler`](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html)
    * [`preprocessing.StandardScaler`](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
  * Support Vector Machines:
    * [`svm.OneClassSVM`](http://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html)
    * [`svm.SVC`](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
    * [`svm.NuSVC`](http://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVC.html)
    * [`svm.SVR`](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html)
    * [`svm.LinearSVR`](http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVR.html)
    * [`svm.NuSVR`](http://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVR.html)
  * Decision Trees:
    * [`tree.DecisionTreeClassifier`](http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
    * [`tree.DecisionTreeRegressor`](http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)
    * [`tree.ExtraTreeClassifier`](http://scikit-learn.org/stable/modules/generated/sklearn.tree.ExtraTreeClassifier.html)
    * [`tree.ExtraTreeRegressor`](http://scikit-learn.org/stable/modules/generated/sklearn.tree.ExtraTreeRegressor.html)
* Supported third-party Estimator and Transformer types:
  * [LightGBM](https://github.com/Microsoft/LightGBM):
    * `lightgbm.LGBMClassifier`
    * `lightgbm.LGBMRegressor`
  * [SkLearn2PMML](https://github.com/jpmml/sklearn2pmml):
    * `sklearn2pmml.EstimatorProxy`
    * `sklearn2pmml.PMMLPipeline`
    * `sklearn2pmml.SelectorProxy`
    * `sklearn2pmml.decoration.CategoricalDomain`
    * `sklearn2pmml.decoration.ContinuousDomain`
    * `sklearn2pmml.preprocessing.ExpressionTransformer`
    * `sklearn2pmml.preprocessing.PMMLLabelBinarizer`
    * `sklearn2pmml.preprocessing.PMMLLabelEncoder`
  * [Sklearn-Pandas](https://github.com/paulgb/sklearn-pandas):
    * `sklearn_pandas.CategoricalImputer`
    * `sklearn_pandas.DataFrameMapper`
  * [XGBoost](https://github.com/dmlc/xgboost):
    * [`xgboost.XGBClassifier`](http://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBClassifier)
    * [`xgboost.XGBRegressor`](http://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBRegressor)
* Production quality:
  * Complete test coverage.
  * Fully compliant with the [JPMML-Evaluator](https://github.com/jpmml/jpmml-evaluator) library.

# Prerequisites #

### The Python side of operations

* Python 2.7, 3.4 or newer.
* [`scikit-learn`](https://pypi.python.org/pypi/scikit-learn) 0.16.0 or newer.
* [`sklearn-pandas`](https://pypi.python.org/pypi/sklearn-pandas) 0.0.10 or newer.
* [`sklearn2pmml`](https://github.com/jpmml/sklearn2pmml) 0.14.0 or newer.

Python installation can be validated as follows:

```python
import sklearn, sklearn.externals.joblib, sklearn_pandas, sklearn2pmml

print(sklearn.__version__)
print(sklearn.externals.joblib.__version__)
print(sklearn_pandas.__version__)
print(sklearn2pmml.__version__)
```

### The JPMML-SkLearn side of operations

* Java 1.7 or newer.

# Installation #

Enter the project root directory and build using [Apache Maven](http://maven.apache.org/):
```
mvn clean install
```

The build produces an executable uber-JAR file `target/converter-executable-1.3-SNAPSHOT.jar`.

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

First, instantiate a `sklearn_pandas.DataFrameMapper` object, which performs **data column-wise** feature engineering and selection work:
```python
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import StandardScaler
from sklearn2pmml.decoration import ContinuousDomain

iris_mapper = DataFrameMapper([
    (["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"], [ContinuousDomain(), StandardScaler()])
])
```

Second, instantiate any number of `Transformer` and `Selector` objects, which perform **dataset-wise** feature engineering and selection work:
```python
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest

iris_pca = PCA(n_components = 3)
iris_selector = SelectKBest(k = 2)
```

Third, instantiate an `Estimator` object:
```python
from sklearn.tree import DecisionTreeClassifier

iris_classifier = DecisionTreeClassifier(min_samples_leaf = 5)
```

Combine the above objects into a `sklearn2pmml.PMMLPipeline` object, and run the experiment:
```python
from sklearn2pmml import PMMLPipeline

iris_pipeline = PMMLPipeline([
    ("mapper", iris_mapper),
    ("pca", iris_pca),
    ("selector", iris_selector),
    ("estimator", iris_classifier)
])
iris_pipeline.fit(iris_df, iris_df["Species"])
```

Store the fitted `sklearn2pmml.PMMLPipeline` object in `pickle` data format:
```python
from sklearn.externals import joblib

joblib.dump(iris_pipeline, "pipeline.pkl.z", compress = 9)
```

Please see the test script file [main.py](https://github.com/jpmml/jpmml-sklearn/blob/master/src/test/resources/main.py) for more classification (binary and multi-class) and regression workflows.

### The JPMML-SkLearn side of operations

Converting the pipeline pickle file `pipeline.pkl.z` to a PMML file `pipeline.pmml`:
```
java -jar target/converter-executable-1.3-SNAPSHOT.jar --pkl-input pipeline.pkl.z --pmml-output pipeline.pmml
```

Getting help:
```
java -jar target/converter-executable-1.3-SNAPSHOT.jar --help
```

# License #

JPMML-SkLearn is licensed under the [GNU Affero General Public License (AGPL) version 3.0](http://www.gnu.org/licenses/agpl-3.0.html). Other licenses are available on request.

# Additional information #

Please contact [info@openscoring.io](mailto:info@openscoring.io)
