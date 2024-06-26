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

import org.dmg.pmml.DataType;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.neural_network.NeuralNetwork;
import org.jpmml.converter.CategoricalLabel;
import org.jpmml.converter.Schema;
import org.jpmml.python.HasArray;
import sklearn.SkLearnClassifier;

public class MLPClassifier extends SkLearnClassifier implements MLPConstants {

	public MLPClassifier(String module, String name){
		super(module, name);
	}

	@Override
	public int getNumberOfFeatures(){
		List<HasArray> coefs = getCoefs();

		return MultilayerPerceptronUtil.getNumberOfFeatures(coefs);
	}

	@Override
	public int getNumberOfOutputs(){
		return 1;
	}

	@Override
	public NeuralNetwork encodeModel(Schema schema){
		String activation = getActivation();

		List<HasArray> coefs = getCoefs();
		List<HasArray> intercepts = getIntercepts();

		CategoricalLabel categoricalLabel = (CategoricalLabel)schema.getLabel();

		NeuralNetwork neuralNetwork = MultilayerPerceptronUtil.encodeNeuralNetwork(MiningFunction.CLASSIFICATION, activation, coefs, intercepts, schema);

		encodePredictProbaOutput(neuralNetwork, DataType.DOUBLE, categoricalLabel);

		return neuralNetwork;
	}

	public String getActivation(){
		return getEnum("activation", this::getString, MLPClassifier.ENUM_ACTIVATION);
	}

	public List<HasArray> getCoefs(){
		return getArrayList("coefs_");
	}

	public List<HasArray> getIntercepts(){
		return getArrayList("intercepts_");
	}
}