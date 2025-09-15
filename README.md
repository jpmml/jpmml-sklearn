JPMML-SkLearn [![Build Status](https://github.com/jpmml/jpmml-sklearn/workflows/maven/badge.svg)](https://github.com/jpmml/jpmml-sklearn/actions?query=workflow%3A%22maven%22)
=============

Java library and command-line application for converting [Scikit-Learn](https://scikit-learn.org/) pipelines to PMML.

# Table of Contents #

* [Features](#features)
  * [Overview](#overview)
  * [Supported packages](#supported-packages)
* [Prerequisites](#prerequisites)
  * [The Python side of operations](#the-python-side-of-operations)
  * [The JPMML-SkLearn side of operations](#the-jpmml-sklearn-side-of-operations)
* [Installation](#installation)
* [Usage](#usage)
  * [The Python side of operations](#the-python-side-of-operations-1)
  * [The JPMML-SkLearn side of operations](#the-jpmml-sklearn-side-of-operations-1)
* [Documentation](#documentation)
* [License](#license)
* [Additional information](#additional-information)

# Features #

### Overview

* Functionality:
  * Three times more supported Python packages, transformers and estimators than all the competitors combined!
  * Thorough collection, analysis and encoding of feature information:
    * Names.
    * Data and operational types.
    * Valid, invalid and missing value spaces.
    * Descriptive statistics.
  * Pipeline extensions:
    * Pruning.
    * Decision engineering (prediction post-processing).
    * Model verification.
  * Conversion options.
* Extensibility:
  * Rich Java APIs for developing custom converters.
  * Automatic discovery and registration of custom converters based on `META-INF/sklearn2pmml.properties` resource files.
  * Direct interfacing with other JPMML conversion libraries such as [JPMML-H2O](https://github.com/jpmml/jpmml-h2o), [JPMML-LightGBM](https://github.com/jpmml/jpmml-lightgbm), [JPMML-StatsModels](https://github.com/jpmml/jpmml-statsmodels) and [JPMML-XGBoost](https://github.com/jpmml/jpmml-xgboost).
* Production quality:
  * Complete test coverage.
  * Fully compliant with the [JPMML-Evaluator](https://github.com/jpmml/jpmml-evaluator) library.

### Supported packages

For a full list of supported transformer and estimator classes see the [`features.md`](features.md) file.

# Prerequisites #

### The Python side of operations

* Python 2.7, 3.4 or newer.
* [`scikit-learn`](https://pypi.python.org/pypi/scikit-learn) 0.16.0 or newer.
* [`sklearn-pandas`](https://pypi.python.org/pypi/sklearn-pandas) 0.0.10 or newer.
* [`sklearn2pmml`](https://github.com/jpmml/sklearn2pmml) 0.14.0 or newer.

Validating Python installation:

```python
import joblib, sklearn, sklearn_pandas, sklearn2pmml

print(joblib.__version__)
print(sklearn.__version__)
print(sklearn_pandas.__version__)
print(sklearn2pmml.__version__)
```

### The JPMML-SkLearn side of operations

* Java 11 or newer.

# Installation #

Enter the project root directory and build using [Apache Maven](https://maven.apache.org/):
```
mvn clean install
```

The build produces a library JAR file `pmml-sklearn/target/pmml-sklearn-1.9-SNAPSHOT.jar`, and an executable uber-JAR file `pmml-sklearn-example/target/pmml-sklearn-example-executable-1.9-SNAPSHOT.jar`.

# Usage #

A typical workflow can be summarized as follows:

1. Use Python to train a model.
2. Serialize the model in `pickle` data format to a file in a local filesystem.
3. Use the JPMML-SkLearn command-line converter application to turn the pickle file to a PMML file.

### The Python side of operations

Loading data to a `pandas.DataFrame` object:

```python
import pandas

df = pandas.read_csv("Iris.csv")

iris_X = df[df.columns.difference(["Species"])]
iris_y = df["Species"]
```

First, creating a `sklearn_pandas.DataFrameMapper` object, which performs **column-oriented** feature engineering and selection work:

```python
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import StandardScaler
from sklearn2pmml.decoration import ContinuousDomain

column_preprocessor = DataFrameMapper([
    (["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"], [ContinuousDomain(), StandardScaler()])
])
```

Second, creating `Transformer` and `Selector` objects, which perform **table-oriented** feature engineering and selection work:

```python
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline
from sklearn2pmml import SelectorProxy

table_preprocessor = Pipeline([
    ("pca", PCA(n_components = 3)),
    ("selector", SelectorProxy(SelectKBest(k = 2)))
])
```

Please note that stateless Scikit-Learn selector objects need to be wrapped into an `sklearn2pmml.SelectprProxy` object.

Third, creating an `Estimator` object:

```python
from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier(min_samples_leaf = 5)
```

Combining the above objects into a `sklearn2pmml.pipeline.PMMLPipeline` object, and running the experiment:

```python
from sklearn2pmml.pipeline import PMMLPipeline

pipeline = PMMLPipeline([
    ("columns", column_preprocessor),
    ("table", table_preprocessor),
    ("classifier", classifier)
])
pipeline.fit(iris_X, iris_y)
```

Recording feature importance information in a `pickle` data format-compatible manner:

```python
classifier.pmml_feature_importances_ = classifier.feature_importances_
```

Embedding model verification data:

```python
pipeline.verify(iris_X.sample(n = 15))
```

Storing the fitted `PMMLPipeline` object in `pickle` data format:

```python
import joblib

joblib.dump(pipeline, "pipeline.pkl.z", compress = 9)
```

Please see the test script file [main.py](https://github.com/jpmml/jpmml-sklearn/blob/master/pmml-sklearn/src/test/resources/main.py) for more classification (binary and multi-class) and regression workflows.

### The JPMML-SkLearn side of operations

Converting the pipeline pickle file `pipeline.pkl.z` to a PMML file `pipeline.pmml`:
```
java -jar pmml-sklearn-example/target/pmml-sklearn-example-executable-1.9-SNAPSHOT.jar --pkl-input pipeline.pkl.z --pmml-output pipeline.pmml
```

Getting help:
```
java -jar pmml-sklearn-example/target/pmml-sklearn-example-executable-1.9-SNAPSHOT.jar --help
```

# Documentation #

Integrations:

* [Training Scikit-Learn GridSearchCV StatsModels pipelines](https://openscoring.io/blog/2023/10/15/sklearn_statsmodels_gridsearchcv_pipeline/)
* [Converting Scikit-Learn H2O.ai pipelines to PMML](https://openscoring.io/blog/2023/07/17/converting_sklearn_h2o_pipeline_pmml/)
* [Converting customized Scikit-Learn estimators to PMML](https://openscoring.io/blog/2023/05/03/converting_sklearn_subclass_pmml/)
* [Training Scikit-Learn StatsModels pipelines](https://openscoring.io/blog/2023/03/28/sklearn_statsmodels_pipeline/)
* [Upgrading Scikit-Learn XGBoost pipelines](https://openscoring.io/blog/2023/02/06/upgrading_sklearn_xgboost_pipeline_pmml/)
* [Training Python-based XGBoost accelerated failure time models](https://openscoring.io/blog/2023/01/28/python_xgboost_aft_pmml/)
* [Converting Scikit-Learn PyCaret 3 pipelines to PMML](https://openscoring.io/blog/2023/01/12/converting_sklearn_pycaret3_pipeline_pmml/)
* [Training Scikit-Learn H2O.ai pipelines](https://openscoring.io/blog/2022/11/11/sklearn_h2o_pipeline/)
* [One-hot encoding categorical features in Scikit-Learn XGBoost pipelines](https://openscoring.io/blog/2022/04/12/onehot_encoding_sklearn_xgboost_pipeline/)
* [Training Scikit-Learn TF(-IDF) plus XGBoost pipelines](https://openscoring.io/blog/2021/02/27/sklearn_tf_tfidf_xgboost_pipeline/)
* [Converting Scikit-Learn TF(-IDF) pipelines to PMML](https://openscoring.io/blog/2021/01/17/converting_sklearn_tf_tfidf_pipeline_pmml/)
* [Converting Scikit-Learn Imbalanced-Learn pipelines to PMML](https://openscoring.io/blog/2020/10/24/converting_sklearn_imblearn_pipeline_pmml/)
* [Converting logistic regression models to PMML](https://openscoring.io/blog/2020/01/19/converting_logistic_regression_pmml/#scikit-learn)
* [Stacking Scikit-Learn, LightGBM and XGBoost models](https://openscoring.io/blog/2020/01/02/stacking_sklearn_lightgbm_xgboost/)
* [Converting Scikit-Learn GridSearchCV pipelines to PMML](https://openscoring.io/blog/2019/12/25/converting_sklearn_gridsearchcv_pipeline_pmml/)
* [Converting Scikit-Learn TPOT pipelines to PMML](https://openscoring.io/blog/2019/06/10/converting_sklearn_tpot_pipeline_pmml/)
* [Converting Scikit-Learn LightGBM pipelines to PMML](https://openscoring.io/blog/2019/04/07/converting_sklearn_lightgbm_pipeline_pmml/)

Extensions:

* [Extending Scikit-Learn with feature cross-references](https://openscoring.io/blog/2023/11/25/sklearn_feature_cross_references/)
* [Extending Scikit-Learn with UDF expression transformer](https://openscoring.io/blog/2023/03/09/sklearn_udf_expression_transformer/)
* [Extending Scikit-Learn with CHAID models](https://openscoring.io/blog/2022/07/14/sklearn_chaid_pmml/)
* [Extending Scikit-Learn with prediction post-processing](https://openscoring.io/blog/2022/05/06/sklearn_prediction_postprocessing/)
* [Extending Scikit-Learn with outlier detector transformer](https://openscoring.io/blog/2021/07/16/sklearn_outlier_detector_transformer/)
* [Extending Scikit-Learn with date and datetime features](https://openscoring.io/blog/2020/03/08/sklearn_date_datetime_pmml/)
* [Extending Scikit-Learn with feature specifications](https://openscoring.io/blog/2020/02/23/sklearn_feature_specification_pmml/)
* [Extending Scikit-Learn with GBDT+LR ensemble models](https://openscoring.io/blog/2019/06/19/sklearn_gbdt_lr_ensemble/)
* [Extending Scikit-Learn with business rules model](https://openscoring.io/blog/2018/09/17/sklearn_business_rules/)

Miscellaneous:

* [Upgrading Scikit-Learn decision tree models](https://openscoring.io/blog/2023/12/29/upgrading_sklearn_decision_tree/)
* [Measuring the memory consumption of Scikit-Learn models](https://openscoring.io/blog/2022/11/09/measuring_memory_sklearn/)
* [Benchmarking Scikit-Learn against JPMML-Evaluator](https://openscoring.io/blog/2021/08/04/benchmarking_sklearn_jpmml_evaluator/)
* [Analyzing Scikit-Learn feature importances via PMML](https://openscoring.io/blog/2021/07/11/analyzing_sklearn_feature_importances_pmml/)

Archived:

* [Converting Scikit-Learn to PMML](https://www.slideshare.net/VilluRuusmann/converting-scikitlearn-to-pmml)

# License #

JPMML-SkLearn is licensed under the terms and conditions of the [GNU Affero General Public License, Version 3.0](https://www.gnu.org/licenses/agpl-3.0.html).

If you would like to use JPMML-SkLearn in a proprietary software project, then it is possible to enter into a licensing agreement which makes JPMML-SkLearn available under the terms and conditions of the [BSD 3-Clause License](https://opensource.org/licenses/BSD-3-Clause) instead.

# Additional information #

JPMML-SkLearn is developed and maintained by Openscoring Ltd, Estonia.

Interested in using [Java PMML API](https://github.com/jpmml) software in your company? Please contact [info@openscoring.io](mailto:info@openscoring.io)
