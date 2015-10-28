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

import com.google.common.base.Function;
import com.google.common.collect.Lists;
import numpy.core.NDArrayUtil;
import org.dmg.pmml.BayesInput;
import org.dmg.pmml.BayesInputs;
import org.dmg.pmml.BayesOutput;
import org.dmg.pmml.DataField;
import org.dmg.pmml.GaussianDistribution;
import org.dmg.pmml.MiningFunctionType;
import org.dmg.pmml.MiningSchema;
import org.dmg.pmml.NaiveBayesModel;
import org.dmg.pmml.Output;
import org.dmg.pmml.TargetValueCount;
import org.dmg.pmml.TargetValueCounts;
import org.dmg.pmml.TargetValueStat;
import org.dmg.pmml.TargetValueStats;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.sklearn.ClassDictUtil;
import sklearn.Classifier;

public class GaussianNB extends Classifier {

	public GaussianNB(String module, String name){
		super(module, name);
	}

	@Override
	public int getNumberOfFeatures(){
		int[] shape = getThetaShape();

		if(shape.length != 2){
			throw new IllegalArgumentException();
		}

		return shape[1];
	}

	@Override
	public NaiveBayesModel encodeModel(List<DataField> dataFields){
		int[] shape = getThetaShape();

		int numberOfClasses = shape[0];
		int numberOfFeatures = shape[1];

		Function<Object, String> targetCategoryFunction = new Function<Object, String>(){

			@Override
			public String apply(Object object){
				return String.valueOf(object);
			}
		};

		List<String> targetCategories = Lists.transform(getClasses(), targetCategoryFunction);

		List<? extends Number> theta = getTheta();
		List<? extends Number> sigma = getSigma();

		BayesInputs bayesInputs = new BayesInputs();

		for(int i = 0; i < numberOfFeatures; i++){
			DataField dataField = dataFields.get(i + 1);

			List<? extends Number> means = NDArrayUtil.getColumn(theta, numberOfClasses, numberOfFeatures, i);
			List<? extends Number> variances = NDArrayUtil.getColumn(sigma, numberOfClasses, numberOfFeatures, i);

			BayesInput bayesInput = new BayesInput(dataField.getName())
				.setTargetValueStats(encodeTargetValueStats(targetCategories, means, variances));

			bayesInputs.addBayesInputs(bayesInput);
		}

		DataField dataField = dataFields.get(0);

		List<? extends Number> classCount = getClassCount();

		BayesOutput bayesOutput = new BayesOutput(dataField.getName(), null)
			.setTargetValueCounts(encodeTargetValueCounts(targetCategories, classCount));

		Output output = new Output(PMMLUtil.createProbabilityFields(dataField));

		MiningSchema miningSchema = PMMLUtil.createMiningSchema(dataFields);

		NaiveBayesModel naiveBayesModel = new NaiveBayesModel(0d, MiningFunctionType.CLASSIFICATION, miningSchema, bayesInputs, bayesOutput)
			.setOutput(output);

		return naiveBayesModel;
	}

	public List<? extends Number> getClassCount(){
		return (List)ClassDictUtil.getArray(this, "class_count_");
	}

	public List<? extends Number> getTheta(){
		return (List)ClassDictUtil.getArray(this, "theta_");
	}

	public List<? extends Number> getSigma(){
		return (List)ClassDictUtil.getArray(this, "sigma_");
	}

	private int[] getThetaShape(){
		return ClassDictUtil.getShape(this, "theta_");
	}

	static
	private TargetValueStats encodeTargetValueStats(List<String> targetCategories, List<? extends Number> means, List<? extends Number> variances){

		if((targetCategories.size() != means.size()) || (targetCategories.size() != variances.size())){
			throw new IllegalArgumentException();
		}

		TargetValueStats targetValueStats = new TargetValueStats();

		for(int i = 0; i < targetCategories.size(); i++){
			Number mean = means.get(i);
			Number variance = variances.get(i);

			GaussianDistribution gaussianDistribution = new GaussianDistribution(mean.doubleValue(), variance.doubleValue());

			TargetValueStat targetValueStat = new TargetValueStat(targetCategories.get(i))
				.setContinuousDistribution(gaussianDistribution);

			targetValueStats.addTargetValueStats(targetValueStat);
		}

		return targetValueStats;
	}

	static
	public TargetValueCounts encodeTargetValueCounts(List<String> targetCategories, List<? extends Number> counts){

		if(targetCategories.size() != counts.size()){
			throw new IllegalArgumentException();
		}

		TargetValueCounts targetValueCounts = new TargetValueCounts();

		for(int i = 0; i < targetCategories.size(); i++){
			Number count = counts.get(i);

			TargetValueCount targetValueCount = new TargetValueCount(targetCategories.get(i), count.intValue());

			targetValueCounts.addTargetValueCounts(targetValueCount);
		}

		return targetValueCounts;
	}
}