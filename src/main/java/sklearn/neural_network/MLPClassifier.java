/*
 * Copyright (c) 2016 Villu Ruusmann
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
package sklearn.neural_network;

import java.util.List;

import numpy.core.NDArray;
import numpy.core.NDArrayUtil;
import org.dmg.pmml.MiningFunctionType;
import org.dmg.pmml.NeuralNetwork;
import org.dmg.pmml.Output;
import org.jpmml.sklearn.Schema;
import sklearn.Classifier;
import sklearn.EstimatorUtil;

public class MLPClassifier extends Classifier {

	public MLPClassifier(String module, String name){
		super(module, name);
	}

	@Override
	public int getNumberOfFeatures(){
		List<?> coefs = getCoefs();

		NDArray input = (NDArray)coefs.get(0);

		int[] shape = NDArrayUtil.getShape(input);

		return shape[0];
	}

	@Override
	public NeuralNetwork encodeModel(Schema schema){
		String activation = getActivation();

		List<?> coefs = getCoefs();
		List<?> intercepts = getIntercepts();

		Output output = EstimatorUtil.encodeClassifierOutput(schema);

		NeuralNetwork neuralNetwork = NeuralNetworkUtil.encodeNeuralNetwork(MiningFunctionType.CLASSIFICATION, activation, coefs, intercepts, schema)
			.setOutput(output);

		return neuralNetwork;
	}

	public String getActivation(){
		return (String)get("activation");
	}

	public List<?> getCoefs(){
		return (List<?>)get("coefs_");
	}

	public List<?> getIntercepts(){
		return (List<?>)get("intercepts_");
	}
}