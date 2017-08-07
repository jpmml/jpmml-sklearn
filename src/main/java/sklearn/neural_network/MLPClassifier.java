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
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.Schema;
import org.jpmml.sklearn.HasArray;
import sklearn.Classifier;

public class MLPClassifier extends Classifier {

	public MLPClassifier(String module, String name){
		super(module, name);
	}

	@Override
	public int getNumberOfFeatures(){
		List<? extends HasArray> coefs = getCoefs();

		return BaseMultilayerPerceptronUtil.getNumberOfFeatures(coefs);
	}

	@Override
	public NeuralNetwork encodeModel(Schema schema){
		String activation = getActivation();

		List<? extends HasArray> coefs = getCoefs();
		List<? extends HasArray> intercepts = getIntercepts();

		NeuralNetwork neuralNetwork = BaseMultilayerPerceptronUtil.encodeNeuralNetwork(MiningFunction.CLASSIFICATION, activation, coefs, intercepts, schema)
			.setOutput(ModelUtil.createProbabilityOutput(DataType.DOUBLE, (CategoricalLabel)schema.getLabel()));

		return neuralNetwork;
	}

	public String getActivation(){
		return (String)get("activation");
	}

	public List<? extends HasArray> getCoefs(){
		return (List)get("coefs_");
	}

	public List<? extends HasArray> getIntercepts(){
		return (List)get("intercepts_");
	}
}