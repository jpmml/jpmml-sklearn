from sklearn.ensemble.forest import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.externals import joblib
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.preprocessing import Binarizer, Imputer, LabelBinarizer, LabelEncoder, MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn_pandas import DataFrameMapper
from pandas import DataFrame

import numpy
import os
import pandas
import pickle
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

#def store_pkl(obj, name):
#    con = open("pkl/" + name, "wb")
#    pickle.dump(obj, con, protocol = -1)
#    con.close()

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
    ("Employment", LabelBinarizer()),
    ("Education", LabelBinarizer()),
    ("Marital", LabelBinarizer()),
    ("Occupation", LabelBinarizer()),
    ("Income", None),
    ("Gender", LabelEncoder()),
    ("Deductions", LabelEncoder()),
    ("Hours", None),
    ("Adjusted", None)
])

audit = audit_mapper.fit_transform(audit_df)

print(audit.shape)

store_pkl(audit_mapper, "Audit.pkl")

audit_X = audit[:, 0:48]
audit_y = audit[:, 48]

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

audit_gbm = GradientBoostingClassifier(random_state = 13, loss = "exponential", init = None)
audit_gbm.fit(audit_X, audit_y)

store_pkl(audit_gbm, "GradientBoostingAudit.pkl")
store_csv(predict_audit(audit_gbm), "GradientBoostingAudit.csv")

audit_logistic = LogisticRegression()
audit_logistic.fit(audit_X, audit_y)

store_pkl(audit_logistic, "LogisticRegressionAudit.pkl")
store_csv(predict_audit(audit_logistic), "LogisticRegressionAudit.csv")

audit_nb = GaussianNB()
audit_nb.fit(audit_X, audit_y)

store_pkl(audit_nb, "NaiveBayesAudit.pkl")
store_csv(predict_audit(audit_nb), "NaiveBayesAudit.csv")

audit_forest = RandomForestClassifier(random_state = 13, min_samples_leaf = 5)
audit_forest.fit(audit_X, audit_y)

store_pkl(audit_forest, "RandomForestAudit.pkl")
store_csv(predict_audit(audit_forest), "RandomForestAudit.csv")

#
# Multi-class classification
#

iris_df = load_csv("Iris.csv")

print(iris_df.dtypes)

iris_mapper = DataFrameMapper([
    ("Sepal.Length", StandardScaler(with_mean = True, with_std = False)),
    ("Sepal.Width", StandardScaler(with_mean = True, with_std = False)),
    ("Petal.Length", StandardScaler(with_std = False)),
    ("Petal.Width", StandardScaler(with_std = False)),
    ("Species", None)
])

iris = iris_mapper.fit_transform(iris_df)

print(iris.shape)

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

iris_gbm = GradientBoostingClassifier(random_state = 13, init = None, n_estimators = 17)
iris_gbm.fit(iris_X, iris_y)

store_pkl(iris_gbm, "GradientBoostingIris.pkl")
store_csv(predict_iris(iris_gbm), "GradientBoostingIris.csv")

iris_logistic = LogisticRegression()
iris_logistic.fit(iris_X, iris_y)

store_pkl(iris_logistic, "LogisticRegressionIris.pkl")
store_csv(predict_iris(iris_logistic), "LogisticRegressionIris.csv")

iris_nb = GaussianNB()
iris_nb.fit(iris_X, iris_y)

store_pkl(iris_nb, "NaiveBayesIris.pkl")
store_csv(predict_iris(iris_nb), "NaiveBayesIris.csv")

iris_forest = RandomForestClassifier(random_state = 13, min_samples_leaf = 5)
iris_forest.fit(iris_X, iris_y)

store_pkl(iris_forest, "RandomForestIris.pkl")
store_csv(predict_iris(iris_forest), "RandomForestIris.csv")

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

auto_mapper = DataFrameMapper([
    (["cylinders"], OneHotEncoder()),
    ("displacement", None),
    (["horsepower"], Imputer(missing_values = "NaN")),
    (["weight"], MinMaxScaler()),
    ("acceleration", StandardScaler(with_mean = False)),
    ("model_year", None),
    (["origin"], Binarizer(threshold = 1)),
    ("mpg", None)
])

auto = auto_mapper.fit_transform(auto_df)

print(auto.shape)

store_pkl(auto_mapper, "Auto.pkl")

auto_X = auto[:, 0:11]
auto_y = auto[:, 11]

print(auto_X.dtype, auto_y.dtype)

def predict_auto(regressor):
    mpg = DataFrame(regressor.predict(auto_X), columns = ["mpg"])
    return mpg

auto_tree = DecisionTreeRegressor(random_state = 13, min_samples_leaf = 5)
auto_tree.fit(auto_X, auto_y)

store_pkl(auto_tree, "DecisionTreeAuto.pkl")
store_csv(predict_auto(auto_tree), "DecisionTreeAuto.csv")

auto_gbm = GradientBoostingRegressor(random_state = 13, init = None)
auto_gbm.fit(auto_X, auto_y)

store_pkl(auto_gbm, "GradientBoostingAuto.pkl")
store_csv(predict_auto(auto_gbm), "GradientBoostingAuto.csv")

auto_linear = LinearRegression()
auto_linear.fit(auto_X, auto_y)

store_pkl(auto_linear, "LinearRegressionAuto.pkl")
store_csv(predict_auto(auto_linear), "LinearRegressionAuto.csv")

auto_forest = RandomForestRegressor(random_state = 13, min_samples_leaf = 5)
auto_forest.fit(auto_X, auto_y)

store_pkl(auto_forest, "RandomForestAuto.pkl")
store_csv(predict_auto(auto_forest), "RandomForestAuto.csv")