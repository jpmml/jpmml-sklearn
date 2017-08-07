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
import org.jpmml.converter.ValueUtil;
import org.jpmml.sklearn.ClassDictUtil;
import sklearn.Classifier;

public class GaussianNB extends Classifier {

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

		List<? extends Number> theta = getTheta();
		List<? extends Number> sigma = getSigma();

		CategoricalLabel categoricalLabel = (CategoricalLabel)schema.getLabel();

		BayesInputs bayesInputs = new BayesInputs();

		for(int i = 0; i < numberOfFeatures; i++){
			Feature feature = schema.getFeature(i);

			List<? extends Number> means = CMatrixUtil.getColumn(theta, numberOfClasses, numberOfFeatures, i);
			List<? extends Number> variances = CMatrixUtil.getColumn(sigma, numberOfClasses, numberOfFeatures, i);

			ContinuousFeature continuousFeature = feature.toContinuousFeature();

			BayesInput bayesInput = new BayesInput(continuousFeature.getName())
				.setTargetValueStats(encodeTargetValueStats(categoricalLabel.getValues(), means, variances));

			bayesInputs.addBayesInputs(bayesInput);
		}

		List<Integer> classCount = getClassCount();

		BayesOutput bayesOutput = new BayesOutput(categoricalLabel.getName(), null)
			.setTargetValueCounts(encodeTargetValueCounts(categoricalLabel.getValues(), classCount));

		NaiveBayesModel naiveBayesModel = new NaiveBayesModel(0d, MiningFunction.CLASSIFICATION, ModelUtil.createMiningSchema(categoricalLabel), bayesInputs, bayesOutput)
			.setOutput(ModelUtil.createProbabilityOutput(DataType.DOUBLE, categoricalLabel));

		return naiveBayesModel;
	}

	public List<Integer> getClassCount(){
		return ValueUtil.asIntegers((List)ClassDictUtil.getArray(this, "class_count_"));
	}

	public List<? extends Number> getTheta(){
		return (List)ClassDictUtil.getArray(this, "theta_");
	}

	public List<? extends Number> getSigma(){
		return (List)ClassDictUtil.getArray(this, "sigma_");
	}

	private int[] getThetaShape(){
		return ClassDictUtil.getShape(this, "theta_", 2);
	}

	static
	private TargetValueStats encodeTargetValueStats(List<String> values, List<? extends Number> means, List<? extends Number> variances){
		TargetValueStats targetValueStats = new TargetValueStats();

		ClassDictUtil.checkSize(values, means, variances);

		for(int i = 0; i < values.size(); i++){
			GaussianDistribution gaussianDistribution = new GaussianDistribution(ValueUtil.asDouble(means.get(i)), ValueUtil.asDouble(variances.get(i)));

			TargetValueStat targetValueStat = new TargetValueStat(values.get(i))
				.setContinuousDistribution(gaussianDistribution);

			targetValueStats.addTargetValueStats(targetValueStat);
		}

		return targetValueStats;
	}

	static
	private TargetValueCounts encodeTargetValueCounts(List<String> values, List<Integer> counts){
		TargetValueCounts targetValueCounts = new TargetValueCounts();

		ClassDictUtil.checkSize(values, counts);

		for(int i = 0; i < values.size(); i++){
			TargetValueCount targetValueCount = new TargetValueCount(values.get(i), counts.get(i));

			targetValueCounts.addTargetValueCounts(targetValueCount);
		}

		return targetValueCounts;
	}
}