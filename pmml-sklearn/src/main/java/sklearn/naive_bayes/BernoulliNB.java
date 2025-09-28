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

import org.dmg.pmml.naive_bayes.BayesInput;
import org.dmg.pmml.naive_bayes.BayesInputs;
import org.dmg.pmml.naive_bayes.NaiveBayesModel;
import org.dmg.pmml.naive_bayes.PairCounts;
import org.jpmml.converter.CMatrixUtil;
import org.jpmml.converter.CategoricalFeature;
import org.jpmml.converter.CategoricalLabel;
import org.jpmml.converter.Feature;
import org.jpmml.converter.Schema;
import org.jpmml.converter.SchemaUtil;

public class BernoulliNB extends DiscreteNB {

	public BernoulliNB(String module, String name){
		super(module, name);
	}

	@Override
	public int getNumberOfFeatures(){
		int[] shape = getFeatureCountShape();

		return shape[1];
	}

	@Override
	public BayesInputs encodeBayesInputs(List<?> values, List<? extends Feature> features){
		int[] shape = getFeatureCountShape();

		int numberOfClasses = shape[0];
		int numberOfFeatures = shape[1];

		Number alpha = getAlpha();
		List<Integer> classCount = getClassCount();
		List<Integer> featureCount = getFeatureCount();

		BayesInputs bayesInputs = new BayesInputs();

		for(int i = 0; i < numberOfFeatures; i++){
			Feature feature = features.get(i);

			List<Integer> featureClassCount = CMatrixUtil.getColumn(featureCount, numberOfClasses, numberOfFeatures, i);

			BayesInput bayesInput;

			if(feature instanceof CategoricalFeature){
				CategoricalFeature categoricalFeature = (CategoricalFeature)feature;

				SchemaUtil.checkSize(2, categoricalFeature);

				List<?> featureValues = categoricalFeature.getValues();

				List<Number> nonEventCounts = new ArrayList<>();
				List<Number> eventCounts = new ArrayList<>();

				for(int j = 0; j < numberOfClasses; j++){
					Number nonEventCount = classCount.get(j) - featureClassCount.get(j);
					Number eventCount = featureClassCount.get(j);

					nonEventCounts.add(nonEventCount);
					eventCounts.add(eventCount);
				}

				List<PairCounts> pairCounts = new ArrayList<>();
				pairCounts.add(DiscreteNBUtil.encodePairCounts(featureValues.get(0), values, alpha, nonEventCounts));
				pairCounts.add(DiscreteNBUtil.encodePairCounts(featureValues.get(1), values, alpha, eventCounts));

				bayesInput = new BayesInput(categoricalFeature.getName(), null, pairCounts);
			} else

			{
				throw new IllegalArgumentException("Expected a categorical feature, got " + feature);
			}

			bayesInputs.addBayesInputs(bayesInput);
		}

		return bayesInputs;
	}

	@Override
	public NaiveBayesModel encodeModel(Schema schema){
		CategoricalLabel categoricalLabel = (CategoricalLabel)schema.getLabel();

		SchemaUtil.checkSize(2, categoricalLabel);

		return super.encodeModel(schema);
	}

	public List<Integer> getFeatureCount(){
		return getIntegerArray("feature_count_");
	}

	public int[] getFeatureCountShape(){
		return getArrayShape("feature_count_", 2);
	}
}