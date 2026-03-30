/*
 * Copyright (c) 2026 Villu Ruusmann
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
import org.dmg.pmml.regression.RegressionModel;
import org.dmg.pmml.regression.RegressionTable;
import org.jpmml.converter.CategoricalLabel;
import org.jpmml.converter.Feature;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.Schema;
import org.jpmml.converter.regression.RegressionModelUtil;
import sklearn.SkLearnClassifier;

abstract
public class ContinuousNB extends SkLearnClassifier {

	public ContinuousNB(String module, String name){
		super(module, name);
	}

	abstract
	public List<Number> getCoefficients(List<Number> featureLogProb, int index, int numberOfClasses, int numberOfFeatures);

	@Override
	public int getNumberOfFeatures(){
		int[] shape = getFeatureCountShape();

		return shape[1];
	}

	@Override
	public RegressionModel encodeModel(Schema schema){
		int[] shape = getFeatureCountShape();

		int numberOfClasses = shape[0];
		int numberOfFeatures = shape[1];

		List<Number> classLogPrior = getClassLogPrior();
		List<Number> featureLogProb = getFeatureLogProb();

		CategoricalLabel categoricalLabel = schema.requireCategoricalLabel()
			.expectCardinality(numberOfClasses);
		List<? extends Feature> features = schema.getFeatures();

		List<RegressionTable> regressionTables = new ArrayList<>();

		for(int i = 0; i < numberOfClasses; i++){
			List<Number> coefficients = getCoefficients(featureLogProb, i, numberOfClasses, numberOfFeatures);
			Number intercept = classLogPrior.get(i);

			RegressionTable regressionTable = RegressionModelUtil.createRegressionTable(features, coefficients, intercept)
				.setTargetCategory(categoricalLabel.getValue(i));

			regressionTables.add(regressionTable);
		}

		RegressionModel regressionModel = new RegressionModel(MiningFunction.CLASSIFICATION, ModelUtil.createMiningSchema(categoricalLabel), regressionTables)
			.setNormalizationMethod(RegressionModel.NormalizationMethod.SOFTMAX);

		encodePredictProbaOutput(regressionModel, DataType.DOUBLE, categoricalLabel);

		return regressionModel;
	}

	public List<Number> getClassLogPrior(){
		return getNumberArray("class_log_prior_");
	}

	public int[] getFeatureCountShape(){
		return getArrayShape("feature_count_", 2);
	}

	public List<Number> getFeatureLogProb(){
		return getNumberArray("feature_log_prob_");
	}
}