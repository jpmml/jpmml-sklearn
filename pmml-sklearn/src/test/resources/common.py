import dill
import joblib
import os
import pandas

def load_csv(name):
	return pandas.read_csv("csv/" + name + ".csv", na_values = ["N/A", "NA"])

def split_csv(df):
	columns = df.columns.tolist()
	return (df[columns[: -1]], df[columns[-1]])

def split_multi_csv(df, target_columns):
	columns = df.columns.tolist()
	feature_columns = [column for column in columns if column not in target_columns]
	return (df[feature_columns], df[target_columns])

def store_csv(df, name):
	df.to_csv("csv/" + name + ".csv", index = False, na_rep = "N/A")

def store_mojo(estimator, name):
	mojo_path = "mojo/" + name + ".zip"
	estimator.download_mojo(mojo_path)
	estimator._mojo_path = mojo_path

def embed_stored_mojo(estimator, maxsize = None):
	mojo_path = estimator._mojo_path
	if (maxsize is None) or (os.path.getsize(mojo_path) <= maxsize):
		with open(mojo_path, "rb") as mojo_file:
			estimator._mojo_bytes = mojo_file.read()

# Joblib dump
def store_pkl(obj, name, flavour = "joblib"):
	path = "pkl/" + name + ".pkl"
	if flavour == "joblib":
		joblib.dump(obj, path, compress = 9)
	elif flavour == "dill":
		with open(path, "wb") as dill_file:
			dill.dump(obj, dill_file)
	else:
		raise ValueError(flavour)

def load_audit(name, stringify = True):
	df = load_csv(name)
	print(df.dtypes)
	df["Adjusted"] = df["Adjusted"].astype(int)
	if stringify:
		df["Deductions"] = df["Deductions"].replace(True, "TRUE").replace(False, "FALSE").astype(str)
	print(df.dtypes)
	return df

def load_auto(name):
	df = load_csv(name)
	print(df.dtypes)
	df["acceleration"] = df["acceleration"].astype(float)
	df["displacement"] = df["displacement"].astype(float)
	df["horsepower"] = df["horsepower"].astype(float)
	df["weight"] = df["weight"].astype(float)
	print(df.dtypes)
	return df

def load_housing(name):
	df = load_csv(name)
	print(df.dtypes)
	df["CHAS"] = df["CHAS"].astype(float)
	df["RAD"] = df["RAD"].astype(float)
	print(df.dtypes)
	return df

def load_iris(name):
	df = load_csv(name)
	print(df.dtypes)
	return df

def load_sentiment(name):
	df = load_csv(name)
	print(df.dtypes)
	return df

def load_versicolor(name):
	df = load_csv(name)
	print(df.dtypes)
	df["Species"] = df["Species"].astype(int)
	return df

def load_visit(name):
	df = load_csv(name)
	print(df.dtypes)
	return df

def load_wheat(name):
	df = load_csv(name)
	print(df.dtypes)
	return df

def load_wine(name):
	df = load_csv(name)
	print(df.dtypes)
	return df