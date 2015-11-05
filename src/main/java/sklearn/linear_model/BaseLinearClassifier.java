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
package sklearn.linear_model;

import java.util.ArrayList;
import java.util.List;

import com.google.common.base.Function;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import numpy.core.NDArrayUtil;
import org.dmg.pmml.DataType;
import org.dmg.pmml.FeatureType;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.FieldRef;
import org.dmg.pmml.MiningModel;
import org.dmg.pmml.OpType;
import org.dmg.pmml.Output;
import org.dmg.pmml.OutputField;
import org.dmg.pmml.PMML;
import org.dmg.pmml.RegressionModel;
import org.dmg.pmml.TransformationDictionary;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.sklearn.ClassDictUtil;
import org.jpmml.sklearn.Schema;
import sklearn.Classifier;
import sklearn.EstimatorUtil;

abstract
public class BaseLinearClassifier extends Classifier {

	public BaseLinearClassifier(String module, String name){
		super(module, name);
	}

	@Override
	public int getNumberOfFeatures(){
		int[] shape = getCoefShape();

		if(shape.length != 2){
			throw new IllegalArgumentException();
		}

		return shape[1];
	}

	@Override
	public MiningModel encodeModel(Schema schema){
		int[] shape = getCoefShape();

		int numberOfClasses = shape[0];
		int numberOfFeatures = shape[1];

		List<String> targetCategories = schema.getTargetCategories();

		List<? extends Number> coefficients = getCoef();
		List<? extends Number> intercepts = getIntercept();

		Function<RegressionModel, FieldName> probabilityFieldFunction = new Function<RegressionModel, FieldName>(){

			@Override
			public FieldName apply(RegressionModel regressionModel){
				Output output = regressionModel.getOutput();

				OutputField outputField = Iterables.getLast(output.getOutputFields());

				return outputField.getName();
			}
		};

		if(numberOfClasses == 1){

			if(targetCategories.size() != 2){
				throw new IllegalArgumentException();
			}

			targetCategories = Lists.reverse(targetCategories);

			RegressionModel regressionModel = encodeCategoryRegressor(targetCategories.get(0), NDArrayUtil.getRow(coefficients, numberOfClasses, numberOfFeatures, 0), intercepts.get(0), schema);

			List<FieldName> probabilityFields = new ArrayList<>();
			probabilityFields.add(probabilityFieldFunction.apply(regressionModel));
			probabilityFields.add(FieldName.create("logitDecisionFunction_" + targetCategories.get(1)));

			return EstimatorUtil.encodeBinomialClassifier(targetCategories, probabilityFields, regressionModel, schema);
		} else

		if(numberOfClasses >= 2){

			if(targetCategories.size() != numberOfClasses){
				throw new IllegalArgumentException();
			}

			List<RegressionModel> regressionModels = new ArrayList<>();

			for(int i = 0; i < targetCategories.size(); i++){
				RegressionModel regressionModel = encodeCategoryRegressor(targetCategories.get(i), NDArrayUtil.getRow(coefficients, numberOfClasses, numberOfFeatures, i), intercepts.get(i), schema);

				regressionModels.add(regressionModel);
			}

			List<FieldName> probabilityFields = Lists.transform(regressionModels, probabilityFieldFunction);

			return EstimatorUtil.encodeMultinomialClassifier(targetCategories, probabilityFields, regressionModels, schema);
		} else

		{
			throw new IllegalArgumentException();
		}
	}

	@Override
	public PMML encodePMML(){
		PMML pmml = super.encodePMML();

		TransformationDictionary transformationDictionary = new TransformationDictionary()
			.addDefineFunctions(EstimatorUtil.encodeLogitFunction());

		pmml.setTransformationDictionary(transformationDictionary);

		return pmml;
	}

	public List<? extends Number> getCoef(){
		return (List)ClassDictUtil.getArray(this, "coef_");
	}

	public List<? extends Number> getIntercept(){
		return (List)ClassDictUtil.getArray(this, "intercept_");
	}

	private int[] getCoefShape(){
		return ClassDictUtil.getShape(this, "coef_");
	}

	static
	private RegressionModel encodeCategoryRegressor(String targetCategory, List<? extends Number> coefficients, Number intercept, Schema schema){
		OutputField decisionFunction = PMMLUtil.createPredictedField(FieldName.create("decisionFunction_" + targetCategory));

		OutputField transformedDecisionFunction = new OutputField(FieldName.create("logitDecisionFunction_" + targetCategory))
			.setFeature(FeatureType.TRANSFORMED_VALUE)
			.setDataType(DataType.DOUBLE)
			.setOpType(OpType.CONTINUOUS)
			.setExpression(PMMLUtil.createApply("logit", new FieldRef(decisionFunction.getName())));

		Output output = new Output()
			.addOutputFields(decisionFunction, transformedDecisionFunction);

		RegressionModel regressionModel = RegressionModelUtil.encodeRegressionModel(coefficients, intercept, schema, false)
			.setOutput(output);

		return regressionModel;
	}
}