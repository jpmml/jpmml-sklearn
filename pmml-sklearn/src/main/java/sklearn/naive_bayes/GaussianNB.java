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
package sklearn.naive_bayes;

import java.util.List;

import org.dmg.pmml.DataType;
import org.dmg.pmml.GaussianDistribution;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.naive_bayes.BayesInput;
import org.dmg.pmml.naive_bayes.BayesInputs;
import org.dmg.pmml.naive_bayes.BayesOutput;
import org.dmg.pmml.naive_bayes.NaiveBayesModel;
import org.dmg.pmml.naive_bayes.TargetValueCount;
import org.dmg.pmml.naive_bayes.TargetValueCounts;
import org.dmg.pmml.naive_bayes.TargetValueStat;
import org.dmg.pmml.naive_bayes.TargetValueStats;
import org.jpmml.converter.CMatrixUtil;
import org.jpmml.converter.CategoricalLabel;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.Feature;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.Schema;
import org.jpmml.python.ClassDictUtil;
import sklearn.SkLearnClassifier;

public class GaussianNB extends SkLearnClassifier {

	public GaussianNB(String module, String name){
		super(module, name);
	}

	@Override
	public int getNumberOfFeatures(){
		int[] shape = getThetaShape();

		return shape[1];
	}

	@Override
	public NaiveBayesModel encodeModel(Schema schema){
		int[] shape = getThetaShape();

		int numberOfClasses = shape[0];
		int numberOfFeatures = shape[1];

		List<Number> theta = getTheta();
		List<Number> sigma = getSigma();

		CategoricalLabel categoricalLabel = (CategoricalLabel)schema.getLabel();

		BayesInputs bayesInputs = new BayesInputs();

		for(int i = 0; i < numberOfFeatures; i++){
			Feature feature = schema.getFeature(i);

			List<Number> means = CMatrixUtil.getColumn(theta, numberOfClasses, numberOfFeatures, i);
			List<Number> variances = CMatrixUtil.getColumn(sigma, numberOfClasses, numberOfFeatures, i);

			ContinuousFeature continuousFeature = feature.toContinuousFeature();

			BayesInput bayesInput = new BayesInput(continuousFeature.getName(), encodeTargetValueStats(categoricalLabel.getValues(), means, variances), null);

			bayesInputs.addBayesInputs(bayesInput);
		}

		List<Integer> classCount = getClassCount();

		BayesOutput bayesOutput = new BayesOutput(null)
			.setTargetField(categoricalLabel.getName())
			.setTargetValueCounts(encodeTargetValueCounts(categoricalLabel.getValues(), classCount));

		NaiveBayesModel naiveBayesModel = new NaiveBayesModel(0d, MiningFunction.CLASSIFICATION, ModelUtil.createMiningSchema(categoricalLabel), bayesInputs, bayesOutput);

		encodePredictProbaOutput(naiveBayesModel, DataType.DOUBLE, categoricalLabel);

		return naiveBayesModel;
	}

	public List<Integer> getClassCount(){
		return getIntegerArray("class_count_");
	}

	public List<Number> getTheta(){
		return getNumberArray("theta_");
	}

	public int[] getThetaShape(){
		return getArrayShape("theta_", 2);
	}

	public List<Number> getSigma(){

		// SkLearn 1.0+
		if(hasattr("var_")){
			return getNumberArray("var_");
		}

		// SkLearn 0.24
		return getNumberArray("sigma_");
	}

	static
	private TargetValueStats encodeTargetValueStats(List<?> values, List<? extends Number> means, List<? extends Number> variances){
		TargetValueStats targetValueStats = new TargetValueStats();

		ClassDictUtil.checkSize(values, means, variances);

		for(int i = 0; i < values.size(); i++){
			GaussianDistribution gaussianDistribution = new GaussianDistribution(means.get(i), variances.get(i));

			TargetValueStat targetValueStat = new TargetValueStat(values.get(i), gaussianDistribution);

			targetValueStats.addTargetValueStats(targetValueStat);
		}

		return targetValueStats;
	}

	static
	private TargetValueCounts encodeTargetValueCounts(List<?> values, List<Integer> counts){
		TargetValueCounts targetValueCounts = new TargetValueCounts();

		ClassDictUtil.checkSize(values, counts);

		for(int i = 0; i < values.size(); i++){
			TargetValueCount targetValueCount = new TargetValueCount(values.get(i), counts.get(i));

			targetValueCounts.addTargetValueCounts(targetValueCount);
		}

		return targetValueCounts;
	}
}