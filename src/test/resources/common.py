from sklearn.externals import joblib

import pandas
#import pickle

def load_csv(name):
	return pandas.read_csv("csv/" + name, na_values = ["N/A", "NA"])

def split_csv(df):
	columns = df.columns.tolist()
	return (df[columns[: -1]], df[columns[-1]])

def store_csv(df, name):
	df.to_csv("csv/" + name, index = False)

# Joblib dump
def store_pkl(obj, name):
	joblib.dump(obj, "pkl/" + name, compress = 9)

# Pickle dump
#def store_pkl(obj, name):
#	con = open("pkl/" + name, "wb")
#	pickle.dump(obj, con, protocol = -1)
#	con.close()

def dump(obj):
	for attr in dir(obj):
		print("obj.%s = %s" % (attr, getattr(obj, attr)))

def load_audit(name):
	df = load_csv(name)
	print(df.dtypes)
	df["Adjusted"] = df["Adjusted"].astype(int)
	df["Deductions"] = df["Deductions"].replace(True, "TRUE").replace(False, "FALSE").astype(str)
	print(df.dtypes)
	return split_csv(df)

def load_auto(name):
	df = load_csv(name)
	print(df.dtypes)
	df["acceleration"] = df["acceleration"].astype(float)
	df["displacement"] = df["displacement"].astype(float)
	df["horsepower"] = df["horsepower"].astype(float)
	df["weight"] = df["weight"].astype(float)
	print(df.dtypes)
	return split_csv(df)

def load_housing(name):
	df = load_csv(name)
	print(df.dtypes)
	df["CHAS"] = df["CHAS"].astype(float)
	df["RAD"] = df["RAD"].astype(float)
	print(df.dtypes)
	return split_csv(df)

def load_iris(name):
	df = load_csv(name)
	print(df.dtypes)
	return split_csv(df)

def load_sentiment(name):
	df = load_csv(name)
	print(df.dtypes)
	return (df["Sentence"], df["Score"])

def load_versicolor(name):
	df = load_csv(name)
	print(df.dtypes)
	df["Species"] = df["Species"].astype(int)
	return split_csv(df)

def load_wheat(name):
	df = load_csv(name)
	print(df.dtypes)
	return split_csv(df)
