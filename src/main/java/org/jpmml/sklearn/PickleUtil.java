/*
 * Copyright (c) 2015 Villu Ruusmann
 *
 * This file is part of JPMML-SkLearn
 *
 * JPMML-SkLearn is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * JPMML-SkLearn is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with JPMML-SkLearn.  If not, see <http://www.gnu.org/licenses/>.
 */
package org.jpmml.sklearn;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;

import joblib.NDArrayWrapperConstructor;
import net.razorvine.pickle.Unpickler;
import numpy.DType;
import numpy.core.NDArray;
import numpy.core.Scalar;
import numpy.random.RandomState;
import sklearn.ensemble.forest.RandomForestClassifier;
import sklearn.ensemble.forest.RandomForestRegressor;
import sklearn.ensemble.gradient_boosting.BinomialDeviance;
import sklearn.ensemble.gradient_boosting.ExponentialLoss;
import sklearn.ensemble.gradient_boosting.GradientBoostingClassifier;
import sklearn.ensemble.gradient_boosting.GradientBoostingRegressor;
import sklearn.ensemble.gradient_boosting.LogOddsEstimator;
import sklearn.ensemble.gradient_boosting.MeanEstimator;
import sklearn.ensemble.gradient_boosting.MultinomialDeviance;
import sklearn.ensemble.gradient_boosting.PriorProbabilityEstimator;
import sklearn.ensemble.gradient_boosting.QuantileEstimator;
import sklearn.ensemble.gradient_boosting.ScaledLogOddsEstimator;
import sklearn.ensemble.gradient_boosting.ZeroEstimator;
import sklearn.linear_model.ElasticNet;
import sklearn.linear_model.Lasso;
import sklearn.linear_model.LinearRegression;
import sklearn.linear_model.LogisticRegression;
import sklearn.linear_model.Ridge;
import sklearn.linear_model.RidgeClassifier;
import sklearn.naive_bayes.GaussianNB;
import sklearn.preprocessing.Binarizer;
import sklearn.preprocessing.Imputer;
import sklearn.preprocessing.LabelBinarizer;
import sklearn.preprocessing.LabelEncoder;
import sklearn.preprocessing.MinMaxScaler;
import sklearn.preprocessing.OneHotEncoder;
import sklearn.preprocessing.StandardScaler;
import sklearn.tree.DecisionTreeClassifier;
import sklearn.tree.DecisionTreeRegressor;
import sklearn.tree.PresortBestSplitter;
import sklearn.tree.RegressionCriterion;
import sklearn.tree.Tree;
import sklearn_pandas.DataFrameMapper;

public class PickleUtil {

	private PickleUtil(){
	}

	static
	public Storage createStorage(File file){
		ZipFileStorage storage = ZipFileStorage.open(file);

		if(storage != null){
			return storage;
		}

		return new FileStorage(file);
	}

	static
	public Object unpickle(Storage storage) throws IOException {
		ObjectConstructor[] constructors = {
			new NDArrayWrapperConstructor("joblib.numpy_pickle", "NDArrayWrapper", storage),
			new ExtensionObjectConstructor("numpy", "dtype", DType.class),
			new ExtensionObjectConstructor("numpy.core.multiarray", "_reconstruct", NDArray.class),
			new ExtensionObjectConstructor("numpy.core.multiarray", "scalar", Scalar.class),
			new ExtensionObjectConstructor("numpy.random", "__RandomState_ctor", RandomState.class),
			new ObjectConstructor("sklearn.ensemble.forest", "RandomForestClassifier", RandomForestClassifier.class),
			new ObjectConstructor("sklearn.ensemble.forest", "RandomForestRegressor", RandomForestRegressor.class),
			new ObjectConstructor("sklearn.ensemble.gradient_boosting", "BinomialDeviance", BinomialDeviance.class),
			new ObjectConstructor("sklearn.ensemble.gradient_boosting", "ExponentialLoss", ExponentialLoss.class),
			new ObjectConstructor("sklearn.ensemble.gradient_boosting", "GradientBoostingClassifier", GradientBoostingClassifier.class),
			new ObjectConstructor("sklearn.ensemble.gradient_boosting", "GradientBoostingRegressor", GradientBoostingRegressor.class),
			new ObjectConstructor("sklearn.ensemble.gradient_boosting", "LogOddsEstimator", LogOddsEstimator.class),
			new ObjectConstructor("sklearn.ensemble.gradient_boosting", "MeanEstimator", MeanEstimator.class),
			new ObjectConstructor("sklearn.ensemble.gradient_boosting", "MultinomialDeviance", MultinomialDeviance.class),
			new ObjectConstructor("sklearn.ensemble.gradient_boosting", "PriorProbabilityEstimator", PriorProbabilityEstimator.class),
			new ObjectConstructor("sklearn.ensemble.gradient_boosting", "QuantileEstimator", QuantileEstimator.class),
			new ObjectConstructor("sklearn.ensemble.gradient_boosting", "ScaledLogOddsEstimator", ScaledLogOddsEstimator.class),
			new ObjectConstructor("sklearn.ensemble.gradient_boosting", "ZeroEstimator", ZeroEstimator.class),
			new NDArrayWrapperConstructor("sklearn.externals.joblib.numpy_pickle", "NDArrayWrapper", storage),
			new ObjectConstructor("sklearn.linear_model.base", "LinearRegression", LinearRegression.class),
			new ObjectConstructor("sklearn.linear_model.coordinate_descent", "ElasticNet", ElasticNet.class),
			new ObjectConstructor("sklearn.linear_model.coordinate_descent", "Lasso", Lasso.class),
			new ObjectConstructor("sklearn.linear_model.logistic", "LogisticRegression", LogisticRegression.class),
			new ObjectConstructor("sklearn.linear_model.ridge", "Ridge", Ridge.class),
			new ObjectConstructor("sklearn.linear_model.ridge", "RidgeClassifier", RidgeClassifier.class),
			new ObjectConstructor("sklearn.naive_bayes", "GaussianNB", GaussianNB.class),
			new ObjectConstructor("sklearn.preprocessing.data", "Binarizer", Binarizer.class),
			new ObjectConstructor("sklearn.preprocessing.data", "MinMaxScaler", MinMaxScaler.class),
			new ObjectConstructor("sklearn.preprocessing.data", "OneHotEncoder", OneHotEncoder.class),
			new ObjectConstructor("sklearn.preprocessing.data", "StandardScaler", StandardScaler.class),
			new ObjectConstructor("sklearn.preprocessing.imputation", "Imputer", Imputer.class),
			new ObjectConstructor("sklearn.preprocessing.label", "LabelBinarizer", LabelBinarizer.class),
			new ObjectConstructor("sklearn.preprocessing.label", "LabelEncoder", LabelEncoder.class),
			new ExtensionObjectConstructor("sklearn.tree._tree", "BestSplitter"),
			new ExtensionObjectConstructor("sklearn.tree._tree", "ClassificationCriterion"),
			new ExtensionObjectConstructor("sklearn.tree._tree", "PresortBestSplitter", PresortBestSplitter.class),
			new ExtensionObjectConstructor("sklearn.tree._tree", "RegressionCriterion", RegressionCriterion.class),
			new ExtensionObjectConstructor("sklearn.tree._tree", "Tree", Tree.class),
			new ObjectConstructor("sklearn.tree.tree", "DecisionTreeClassifier", DecisionTreeClassifier.class),
			new ObjectConstructor("sklearn.tree.tree", "DecisionTreeRegressor", DecisionTreeRegressor.class),
			new ObjectConstructor("sklearn_pandas", "DataFrameMapper", DataFrameMapper.class),
		};

		for(ObjectConstructor constructor : constructors){
			Unpickler.registerConstructor(constructor.getModule(), constructor.getName(), constructor);
		}

		InputStream is = storage.getObject();

		try {
			Unpickler unpickler = new Unpickler();

			return unpickler.load(is);
		} finally {
			is.close();
		}
	}
}