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

import java.util.ArrayList;
import java.util.List;

import org.dmg.pmml.DataType;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.naive_bayes.BayesInput;
import org.dmg.pmml.naive_bayes.BayesInputs;
import org.dmg.pmml.naive_bayes.BayesOutput;
import org.dmg.pmml.naive_bayes.NaiveBayesModel;
import org.dmg.pmml.naive_bayes.PairCounts;
import org.dmg.pmml.naive_bayes.TargetValueCount;
import org.dmg.pmml.naive_bayes.TargetValueCounts;
import org.jpmml.converter.CMatrixUtil;
import org.jpmml.converter.CategoricalLabel;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.Feature;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.Schema;
import org.jpmml.converter.SchemaUtil;
import sklearn.SkLearnClassifier;

public class BernoulliNB extends SkLearnClassifier {

	public BernoulliNB(String module, String name){
		super(module, name);
	}

	@Override
	public int getNumberOfFeatures(){
		int[] shape = getFeatureCountShape();

		return shape[1];
	}

	@Override
	public NaiveBayesModel encodeModel(Schema schema){
		int[] shape = getFeatureCountShape();

		int numberOfClasses = shape[0];
		int numberOfFeatures = shape[1];

		List<Integer> classCount = getClassCount();
		List<Integer> featureCount = getFeatureCount();

		CategoricalLabel categoricalLabel = (CategoricalLabel)schema.getLabel();

		SchemaUtil.checkSize(2, categoricalLabel);

		List<?> values = categoricalLabel.getValues();

		BayesInputs bayesInputs = new BayesInputs();

		for(int i = 0; i < numberOfFeatures; i++){
			Feature feature = schema.getFeature(i);

			List<Integer> featureClassCount = CMatrixUtil.getColumn(featureCount, numberOfClasses, numberOfFeatures, i);

			ContinuousFeature continuousFeature = feature.toContinuousFeature();

			List<Integer> nonEventCounts = new ArrayList<>();

			for(int j = 0; j < numberOfClasses; j++){
				nonEventCounts.add(classCount.get(j) - featureClassCount.get(j));
			}

			List<Integer> eventCounts = new ArrayList<>();
			eventCounts.addAll(featureClassCount);

			List<PairCounts> pairCounts = new ArrayList<>();
			pairCounts.add(encodePairCounts(values.get(0), values, nonEventCounts));
			pairCounts.add(encodePairCounts(values.get(1), values, eventCounts));

			BayesInput bayesInput = new BayesInput(continuousFeature.getName(), null, pairCounts);

			bayesInputs.addBayesInputs(bayesInput);
		}

		BayesOutput bayesOutput = new BayesOutput(null)
			.setTargetField(categoricalLabel.getName())
			.setTargetValueCounts(encodeTargetValueCounts(values, classCount));

		NaiveBayesModel naiveBayesModel = new NaiveBayesModel(0d, MiningFunction.CLASSIFICATION, ModelUtil.createMiningSchema(categoricalLabel), bayesInputs, bayesOutput);

		encodePredictProbaOutput(naiveBayesModel, DataType.DOUBLE, categoricalLabel);

		return naiveBayesModel;
	}

	public List<Integer> getClassCount(){
		return getIntegerArray("class_count_");
	}

	public List<Integer> getFeatureCount(){
		return getIntegerArray("feature_count_");
	}

	public int[] getFeatureCountShape(){
		return getArrayShape("feature_count_", 2);
	}

	static
	private PairCounts encodePairCounts(Object value, List<?> values, List<Integer> counts){
		PairCounts pairCounts = new PairCounts()
			.setValue(value)
			.setTargetValueCounts(encodeTargetValueCounts(values, counts));

		return pairCounts;
	}

	static
	private TargetValueCounts encodeTargetValueCounts(List<?> values, List<Integer> counts){
		TargetValueCounts targetValueCounts = new TargetValueCounts();

		for(int i = 0; i < values.size(); i++){
			Object value = values.get(i);
			Integer count = counts.get(i);

			TargetValueCount targetValueCount = new TargetValueCount(value, count);

			targetValueCounts.addTargetValueCounts(targetValueCount);
		}

		return targetValueCounts;
	}
}