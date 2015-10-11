from sklearn.ensemble.forest import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.preprocessing import Imputer, LabelEncoder, MinMaxScaler, StandardScaler
from sklearn_pandas import DataFrameMapper
from pandas import DataFrame

import joblib
import numpy
import os
import pandas
import shutil
import tempfile
import zipfile

def load_csv(name):
    return pandas.read_csv("csv/" + name, na_values = ["N/A", "NA"])

def store_csv(df, name):
    df.to_csv("csv/" + name, index = False)

def store_pkl(obj, name):
    tmpDir = tempfile.mkdtemp()
    print(tmpDir)
    try:
        tmpFiles = joblib.dump(obj, os.path.join(tmpDir, name), compress = 0)
        # print(tmpFiles)
        zipDir = zipfile.ZipFile("pkl/" + name + ".zip", "w")
        for tmpFile in tmpFiles:
            zipDir.write(tmpFile, os.path.relpath(tmpFile, tmpDir))
        zipDir.close()
        print(zipDir.filename)
    finally:
        shutil.rmtree(tmpDir)

def dump(obj):
    for attr in dir(obj):
        print("obj.%s = %s" % (attr, getattr(obj, attr)))

#
# Binary classification
#

audit_df = load_csv("Audit.csv")

print(audit_df.dtypes)

audit_df["Deductions"] = audit_df["Deductions"].replace(True, "TRUE").replace(False, "FALSE").astype(str)

print(audit_df.dtypes)

audit_mapper = DataFrameMapper([
    ("Age", None),
    ("Employment", LabelEncoder()),
    ("Education", LabelEncoder()),
    ("Marital", LabelEncoder()),
    ("Occupation", LabelEncoder()),
    ("Income", None),
    ("Gender", LabelEncoder()),
    ("Deductions", LabelEncoder()),
    ("Hours", None),
    ("Adjusted", None)
])

audit = audit_mapper.fit_transform(audit_df)

store_pkl(audit_mapper, "Audit.pkl")

audit_X = audit[:, 0:9]
audit_y = audit[:, 9]

audit_y = audit_y.astype(int)

print(audit_X.dtype, audit_y.dtype)

def predict_audit(classifier):
    adjusted = DataFrame(classifier.predict(audit_X), columns = ["Adjusted"])
    adjusted_proba = DataFrame(classifier.predict_proba(audit_X), columns = ["probability_0", "probability_1"])
    return pandas.concat((adjusted, adjusted_proba), axis = 1)

audit_tree = DecisionTreeClassifier(random_state = 13, min_samples_leaf = 5)
audit_tree.fit(audit_X, audit_y)

store_pkl(audit_tree, "DecisionTreeAudit.pkl")
store_csv(predict_audit(audit_tree), "DecisionTreeAudit.csv")

audit_forest = RandomForestClassifier(random_state = 13, min_samples_leaf = 5)
audit_forest.fit(audit_X, audit_y)

store_pkl(audit_forest, "RandomForestAudit.pkl")
store_csv(predict_audit(audit_forest), "RandomForestAudit.csv")

audit_regression = LogisticRegression()
audit_regression.fit(audit_X, audit_y)

store_pkl(audit_regression, "RegressionAudit.pkl")
store_csv(predict_audit(audit_regression), "RegressionAudit.csv")

#
# Multi-class classification
#

iris_df = load_csv("Iris.csv")

print(iris_df.dtypes)

iris_mapper = DataFrameMapper([
    ("Sepal.Length", None),
    ("Sepal.Width", None),
    ("Petal.Length", None),
    ("Petal.Width", None),
    ("Species", None)
])

iris = iris_mapper.fit_transform(iris_df)

store_pkl(iris_mapper, "Iris.pkl")

iris_X = iris[:, 0:4]
iris_y = iris[:, 4]

print(iris_X.dtype, iris_y.dtype)

def predict_iris(classifier):
    species = DataFrame(classifier.predict(iris_X), columns = ["Species"])
    species_proba = DataFrame(classifier.predict_proba(iris_X), columns = ["probability_setosa", "probability_versicolor", "probability_virginica"])
    return pandas.concat((species, species_proba), axis = 1)

iris_tree = DecisionTreeClassifier(random_state = 13, min_samples_leaf = 5)
iris_tree.fit(iris_X, iris_y)

store_pkl(iris_tree, "DecisionTreeIris.pkl")
store_csv(predict_iris(iris_tree), "DecisionTreeIris.csv")

iris_forest = RandomForestClassifier(random_state = 13, min_samples_leaf = 5)
iris_forest.fit(iris_X, iris_y)

store_pkl(iris_forest, "RandomForestIris.pkl")
store_csv(predict_iris(iris_forest), "RandomForestIris.csv")

iris_regression = LogisticRegression()
iris_regression.fit(iris_X, iris_y)

store_pkl(iris_regression, "RegressionIris.pkl")
store_csv(predict_iris(iris_regression), "RegressionIris.csv")

#
# Regression
#

auto_df = load_csv("Auto.csv")

print(auto_df.dtypes)

auto_df["displacement"] = auto_df["displacement"].astype(float)
auto_df["horsepower"] = auto_df["horsepower"].astype(float)
auto_df["weight"] = auto_df["weight"].astype(float)
auto_df["acceleration"] = auto_df["acceleration"].astype(float)

print(auto_df.dtypes)

# See https://github.com/paulgb/sklearn-pandas/issues/9
auto_mapper = DataFrameMapper([
    ("cylinders", None),
    ("displacement", None),
    (["horsepower"], Imputer()),
    (["weight"], MinMaxScaler()),
    ("acceleration", StandardScaler()),
    ("model_year", None),
    ("origin", None),
    ("mpg", None)
])

auto = auto_mapper.fit_transform(auto_df)

store_pkl(auto_mapper, "Auto.pkl")

auto_X = auto[:, 0:7]
auto_y = auto[:, 7]

print(auto_X.dtype, auto_y.dtype)

def predict_auto(regressor):
    mpg = DataFrame(regressor.predict(auto_X), columns = ["mpg"])
    return mpg

auto_tree = DecisionTreeRegressor(random_state = 13, min_samples_leaf = 5)
auto_tree.fit(auto_X, auto_y)

store_pkl(auto_tree, "DecisionTreeAuto.pkl")
store_csv(predict_auto(auto_tree), "DecisionTreeAuto.csv")

auto_forest = RandomForestRegressor(random_state = 13, min_samples_leaf = 5)
auto_forest.fit(auto_X, auto_y)

store_pkl(auto_forest, "RandomForestAuto.pkl")
store_csv(predict_auto(auto_forest), "RandomForestAuto.csv")

auto_regression = LinearRegression()
auto_regression.fit(auto_X, auto_y)

store_pkl(auto_regression, "RegressionAuto.pkl")
store_csv(predict_auto(auto_regression), "RegressionAuto.csv")