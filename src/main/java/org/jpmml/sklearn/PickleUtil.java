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

import joblib.NDArrayWrapper;
import net.razorvine.pickle.Unpickler;
import numpy.DType;
import numpy.core.NDArray;
import numpy.core.Scalar;
import sklearn.ensemble.RandomForestClassifier;
import sklearn.ensemble.RandomForestRegressor;
import sklearn.preprocessing.Imputer;
import sklearn.preprocessing.LabelEncoder;
import sklearn.preprocessing.MinMaxScaler;
import sklearn.preprocessing.StandardScaler;
import sklearn.tree.DecisionTreeClassifier;
import sklearn.tree.DecisionTreeRegressor;
import sklearn.tree.Tree;
import sklearn_pandas.DataFrameMapper;

public class PickleUtil {

	private PickleUtil(){
	}

	static
	public Storage createStorage(File file) throws IOException {

		if(ZipFileStorage.accept(file)){
			return new ZipFileStorage(file);
		}

		return new FileStorage(file);
	}

	static
	public Object unpickle(final Storage storage) throws IOException {
		ObjectConstructor[] constructors = {
			new CClassDictConstructor("joblib.numpy_pickle", "NDArrayWrapper"){

				@Override
				public NDArrayWrapper newObject(){
					return new NDArrayWrapper(getModule(), getName()){

						@Override
						public InputStream getInputStream() throws IOException {
							return storage.getArray(getFileName());
						}
					};
				}
			},
			new CClassDictConstructor("numpy", "dtype", DType.class),
			new CClassDictConstructor("numpy.core.multiarray", "_reconstruct", NDArray.class),
			new CClassDictConstructor("numpy.core.multiarray", "scalar", Scalar.class),
			new CClassDictConstructor("numpy.random", "__RandomState_ctor"),
			new ObjectConstructor("sklearn.ensemble.forest", "RandomForestClassifier", RandomForestClassifier.class),
			new ObjectConstructor("sklearn.ensemble.forest", "RandomForestRegressor", RandomForestRegressor.class),
			new ObjectConstructor("sklearn.preprocessing.data", "MinMaxScaler", MinMaxScaler.class),
			new ObjectConstructor("sklearn.preprocessing.imputation", "Imputer", Imputer.class),
			new ObjectConstructor("sklearn.preprocessing.data", "StandardScaler", StandardScaler.class),
			new ObjectConstructor("sklearn.preprocessing.label", "LabelEncoder", LabelEncoder.class),
			new CClassDictConstructor("sklearn.tree._tree", "BestSplitter"),
			new CClassDictConstructor("sklearn.tree._tree", "ClassificationCriterion"),
			new CClassDictConstructor("sklearn.tree._tree", "RegressionCriterion"),
			new CClassDictConstructor("sklearn.tree._tree", "Tree", Tree.class),
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