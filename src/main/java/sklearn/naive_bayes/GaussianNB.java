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

import org.dmg.pmml.BayesInput;
import org.dmg.pmml.BayesInputs;
import org.dmg.pmml.BayesOutput;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.GaussianDistribution;
import org.dmg.pmml.MiningFunctionType;
import org.dmg.pmml.NaiveBayesModel;
import org.dmg.pmml.TargetValueCount;
import org.dmg.pmml.TargetValueCounts;
import org.dmg.pmml.TargetValueStat;
import org.dmg.pmml.TargetValueStats;
import org.jpmml.converter.Feature;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.Schema;
import org.jpmml.converter.ValueUtil;
import org.jpmml.sklearn.ClassDictUtil;
import org.jpmml.sklearn.MatrixUtil;
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

		List<String> targetCategories = schema.getTargetCategories();

		List<? extends Number> theta = getTheta();
		List<? extends Number> sigma = getSigma();

		BayesInputs bayesInputs = new BayesInputs();

		for(int i = 0; i < numberOfFeatures; i++){
			Feature feature = schema.getFeature(i);

			List<? extends Number> means = MatrixUtil.getColumn(theta, numberOfClasses, numberOfFeatures, i);
			List<? extends Number> variances = MatrixUtil.getColumn(sigma, numberOfClasses, numberOfFeatures, i);

			BayesInput bayesInput = new BayesInput(feature.getName())
				.setTargetValueStats(encodeTargetValueStats(targetCategories, means, variances));

			bayesInputs.addBayesInputs(bayesInput);
		}

		FieldName targetField = schema.getTargetField();
		if(targetField == null){
			throw new IllegalArgumentException();
		}

		List<Integer> classCount = getClassCount();

		BayesOutput bayesOutput = new BayesOutput(targetField, null)
			.setTargetValueCounts(encodeTargetValueCounts(targetCategories, classCount));

		NaiveBayesModel naiveBayesModel = new NaiveBayesModel(0d, MiningFunctionType.CLASSIFICATION, ModelUtil.createMiningSchema(schema), bayesInputs, bayesOutput)
			.setOutput(ModelUtil.createProbabilityOutput(schema));

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
	private TargetValueStats encodeTargetValueStats(List<String> targetCategories, List<? extends Number> means, List<? extends Number> variances){

		if((targetCategories.size() != means.size()) || (targetCategories.size() != variances.size())){
			throw new IllegalArgumentException();
		}

		TargetValueStats targetValueStats = new TargetValueStats();

		for(int i = 0; i < targetCategories.size(); i++){
			GaussianDistribution gaussianDistribution = new GaussianDistribution(ValueUtil.asDouble(means.get(i)), ValueUtil.asDouble(variances.get(i)));

			TargetValueStat targetValueStat = new TargetValueStat(targetCategories.get(i))
				.setContinuousDistribution(gaussianDistribution);

			targetValueStats.addTargetValueStats(targetValueStat);
		}

		return targetValueStats;
	}

	static
	public TargetValueCounts encodeTargetValueCounts(List<String> targetCategories, List<Integer> counts){

		if(targetCategories.size() != counts.size()){
			throw new IllegalArgumentException();
		}

		TargetValueCounts targetValueCounts = new TargetValueCounts();

		for(int i = 0; i < targetCategories.size(); i++){
			TargetValueCount targetValueCount = new TargetValueCount(targetCategories.get(i), counts.get(i));

			targetValueCounts.addTargetValueCounts(targetValueCount);
		}

		return targetValueCounts;
	}
}