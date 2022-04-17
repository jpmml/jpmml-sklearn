from common import *

from pandas import DataFrame
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPRegressor
from sklearn2pmml.decoration import Alias, CategoricalDomain, ContinuousDomain
from sklearn2pmml.neural_network import MLPTransformer
from sklearn2pmml.pipeline import PMMLPipeline
from sklearn2pmml.postprocessing import BusinessDecisionTransformer

mlp = MLPRegressor(hidden_layer_sizes = (11, ), solver = "lbfgs", random_state = 13)

def build_iris(iris_df, name, transformer, predict_proba_transformer = None):
	iris_X, iris_y = split_csv(iris_df)

	pipeline = PMMLPipeline([
		("decorator", ContinuousDomain()),
		("transformer", transformer),
		("classifier", LogisticRegression(random_state = 13))
	], predict_proba_transformer = predict_proba_transformer)
	pipeline.fit(iris_X, iris_y)
	pipeline.verify(iris_X.sample(frac = 0.05, random_state = 13))
	store_pkl(pipeline, name)
	species = DataFrame(pipeline.predict(iris_X), columns = ["Species"])
	if predict_proba_transformer is not None:
		species_proba = DataFrame(pipeline.predict_proba_transform(iris_X), columns = ["probability(setosa)", "probability(versicolor)", "probability(virginica)", "decision"])
	else:
		species_proba = DataFrame(pipeline.predict_proba(iris_X), columns = ["probability(setosa)", "probability(versicolor)", "probability(virginica)"])
	species = pandas.concat((species, species_proba), axis = 1)
	store_csv(species, name)

iris_df = load_iris("Iris")

build_iris(iris_df, "MLPAutoencoderIris", MLPTransformer(mlp), predict_proba_transformer = Alias(BusinessDecisionTransformer("'yes' if X[1] >= 0.95 else 'no'", "Is the predicted species definitely versicolor?", [("yes", "Is versicolor"), ("no", "Is not versicolor")], prefit = True), "decision", prefit = True))
build_iris(iris_df, "MLPTransformerIris", MLPTransformer(mlp, transformer_output_layer = 1))