from sklearn.ensemble.forest import RandomForestClassifier, RandomForestRegressor
from sklearn.tree.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
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
    return pandas.read_csv("csv/" + name)

def store_csv(df, name):
    df.to_csv("csv/" + name, index = False)

def store_pkl(obj, name):
    tmpDir = tempfile.mkdtemp()
    print(tmpDir)
    try:
        tmpFiles = joblib.dump(obj, os.path.join(tmpDir, name), compress = 0)
        print(tmpFiles)
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

audit = load_csv("Audit.csv")

audit["Adjusted"] = audit["Adjusted"].astype(int)

audit_mapper = DataFrameMapper([
    ("Age", None),
    ("Employment", LabelEncoder()),
    ("Education", LabelEncoder()),
    ("Marital", LabelEncoder()),
    ("Occupation", LabelEncoder()),
    ("Income", None),
    ("Gender", LabelEncoder()),
    ("Deductions", None),
    ("Hours", None),
    ("Adjusted", None)
])

audit = audit_mapper.fit_transform(audit)

store_pkl(audit_mapper, "Audit.pkl")

audit_X = audit[:, 0:9]
audit_y = audit[:, 9]

audit_y = audit_y.astype(int)

print(audit_X, audit_y)

def predict_audit(classifier):
    adjusted = DataFrame(classifier.predict(audit_X), columns = ["Adjusted"])
    adjusted_proba = DataFrame(classifier.predict_proba(audit_X), columns = ["probability_0", "probability_1"])
    return pandas.concat((adjusted, adjusted_proba), axis = 1)

audit_tree = DecisionTreeClassifier(min_samples_leaf = 5)
audit_tree.fit(audit_X, audit_y)

store_pkl(audit_tree, "DecisionTreeAudit.pkl")
store_csv(predict_audit(audit_tree), "DecisionTreeAudit.csv")

audit_forest = RandomForestClassifier(random_state = 13, min_samples_leaf = 5)
audit_forest.fit(audit_X, audit_y)

store_pkl(audit_forest, "RandomForestAudit.pkl")
store_csv(predict_audit(audit_forest), "RandomForestAudit.csv")

#
# Multi-class classification
#

iris = load_csv("Iris.csv")

iris_mapper = DataFrameMapper([
    ("Sepal.Length", None),
    ("Sepal.Width", None),
    ("Petal.Length", None),
    ("Petal.Width", None),
    ("Species", None)
])

iris = iris_mapper.fit_transform(iris)

store_pkl(iris_mapper, "Iris.pkl")

iris_X = iris[:, 0:4]
iris_y = iris[:, 4]

print(iris_X, iris_y)

def predict_iris(classifier):
    species = DataFrame(classifier.predict(iris_X), columns = ["Species"])
    species_proba = DataFrame(classifier.predict_proba(iris_X), columns = ["probability_setosa", "probability_versicolor", "probability_virginica"])
    return pandas.concat((species, species_proba), axis = 1)

iris_tree = DecisionTreeClassifier(min_samples_leaf = 5)
iris_tree.fit(iris_X, iris_y)

store_pkl(iris_tree, "DecisionTreeIris.pkl")
store_csv(predict_iris(iris_tree), "DecisionTreeIris.csv")

iris_forest = RandomForestClassifier(random_state = 13, min_samples_leaf = 5)
iris_forest.fit(iris_X, iris_y)

store_pkl(iris_forest, "RandomForestIris.pkl")
store_csv(predict_iris(iris_forest), "RandomForestIris.csv")

#
# Regression
#

auto = load_csv("Auto.csv")

auto["horsepower"] = auto["horsepower"].astype(float)
auto["weight"] = auto["weight"].astype(float)
auto["acceleration"] = auto["acceleration"].astype(float)

# See https://github.com/paulgb/sklearn-pandas/issues/9
auto_mapper = DataFrameMapper([
    ("cylinders", None),
    ("displacement", None),
    ("horsepower", StandardScaler()),
    (["weight"], MinMaxScaler()),
    ("acceleration", StandardScaler()),
    ("model_year", None),
    ("origin", None),
    ("mpg", None)
])

auto = auto_mapper.fit_transform(auto)

store_pkl(auto_mapper, "Auto.pkl")

auto_X = auto[:, 0:7]
auto_y = auto[:, 7]

print(auto_X, auto_y)

def predict_auto(regressor):
    mpg = DataFrame(regressor.predict(auto_X), columns = ["mpg"])
    return mpg

auto_tree = DecisionTreeRegressor(min_samples_leaf = 5)
auto_tree.fit(auto_X, auto_y)

store_pkl(auto_tree, "DecisionTreeAuto.pkl")
store_csv(predict_auto(auto_tree), "DecisionTreeAuto.csv")

auto_forest = RandomForestRegressor(random_state = 13, min_samples_leaf = 5)
auto_forest.fit(auto_X, auto_y)

store_pkl(auto_forest, "RandomForestAuto.pkl")
store_csv(predict_auto(auto_forest), "RandomForestAuto.csv")