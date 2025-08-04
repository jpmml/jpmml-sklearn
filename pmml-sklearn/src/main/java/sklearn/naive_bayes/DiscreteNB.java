/*
 * Copyright (c) 2025 Villu Ruusmann
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
package sklearn.naive_bayes;

import java.util.List;

import org.dmg.pmml.DataType;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.naive_bayes.BayesInputs;
import org.dmg.pmml.naive_bayes.BayesOutput;
import org.dmg.pmml.naive_bayes.NaiveBayesModel;
import org.jpmml.converter.CategoricalLabel;
import org.jpmml.converter.Feature;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.Schema;
import sklearn.SkLearnClassifier;

abstract
public class DiscreteNB extends SkLearnClassifier {

	public DiscreteNB(String module, String name){
		super(module, name);
	}

	abstract
	public BayesInputs encodeBayesInputs(List<?> values, List<? extends Feature> features);

	@Override
	public NaiveBayesModel encodeModel(Schema schema){
		List<Integer> classCount = getClassCount();

		CategoricalLabel categoricalLabel = (CategoricalLabel)schema.getLabel();
		List<? extends Feature> features = schema.getFeatures();

		List<?> values = categoricalLabel.getValues();

		BayesInputs bayesInputs = encodeBayesInputs(values, features);

		BayesOutput bayesOutput = new BayesOutput()
			.setTargetField(categoricalLabel.getName())
			.setTargetValueCounts(DiscreteNBUtil.encodeTargetValueCounts(values, classCount));

		NaiveBayesModel naiveBayesModel = new NaiveBayesModel(0d, MiningFunction.CLASSIFICATION, ModelUtil.createMiningSchema(categoricalLabel), bayesInputs, bayesOutput);

		encodePredictProbaOutput(naiveBayesModel, DataType.DOUBLE, categoricalLabel);

		return naiveBayesModel;
	}

	public Number getAlpha(){
		return getNumber("alpha");
	}

	public List<Integer> getClassCount(){
		return getIntegerArray("class_count_");
	}
}