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

import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import com.google.common.collect.Iterables;
import joblib.NDArrayWrapper;
import numpy.core.NDArray;
import numpy.core.NDArrayUtil;
import org.dmg.pmml.DataField;
import org.dmg.pmml.DataType;
import org.dmg.pmml.DefineFunction;
import org.dmg.pmml.Expression;
import org.dmg.pmml.FeatureType;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.FieldRef;
import org.dmg.pmml.FieldUsageType;
import org.dmg.pmml.MiningField;
import org.dmg.pmml.MiningFunctionType;
import org.dmg.pmml.MiningModel;
import org.dmg.pmml.MiningSchema;
import org.dmg.pmml.MultipleModelMethodType;
import org.dmg.pmml.NumericPredictor;
import org.dmg.pmml.OpType;
import org.dmg.pmml.Output;
import org.dmg.pmml.OutputField;
import org.dmg.pmml.PMML;
import org.dmg.pmml.ParameterField;
import org.dmg.pmml.RegressionModel;
import org.dmg.pmml.RegressionNormalizationMethodType;
import org.dmg.pmml.RegressionTable;
import org.dmg.pmml.Segment;
import org.dmg.pmml.Segmentation;
import org.dmg.pmml.TransformationDictionary;
import org.dmg.pmml.True;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.sklearn.ClassDictUtil;
import sklearn.Classifier;

public class LogisticRegression extends Classifier {

	public LogisticRegression(String module, String name){
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
	public MiningModel encodeModel(List<DataField> dataFields){
		DataField dataField = dataFields.get(0);

		int[] shape = getCoefShape();

		int numberOfClasses = shape[0];
		int numberOfFeatures = shape[1];

		List<? extends Number> coefficients = getCoef();

		List<? extends Number> intercepts = getIntercept();

		List<?> classes = getClasses();

		if(numberOfClasses == 1){

			if(classes.size() != 2){
				throw new IllegalArgumentException();
			}
		} else

		{
			if(classes.size() != numberOfClasses){
				throw new IllegalArgumentException();
			}
		}

		MiningSchema regressionMiningSchema = PMMLUtil.createMiningSchema(null, dataFields.subList(1, dataFields.size()));

		Segmentation segmentation = new Segmentation(MultipleModelMethodType.MODEL_CHAIN, null);

		Map<String, FieldName> categoryProbabilities = new LinkedHashMap<>();

		for(int i = 0; i < numberOfClasses; i++){
			String targetCategory;

			// The probability of the second class is obtained by computation.
			// The probability of the first class is obtained by subtracting the probability of the second class from 1.0
			if(numberOfClasses == 1){
				targetCategory = String.valueOf(classes.get(classes.size() - 1));
			} else

			{
				targetCategory = String.valueOf(classes.get(i));
			}

			RegressionTable regressionTable = RegressionModelUtil.encodeRegressionTable(getRow(coefficients, numberOfClasses, numberOfFeatures, i), intercepts.get(i), dataFields);

			OutputField decisionFunction = PMMLUtil.createPredictedField(FieldName.create("decisionFunction_" + targetCategory));

			OutputField logitDecisionFunction = new OutputField(FieldName.create("logitDecisionFunction_" + targetCategory))
				.setFeature(FeatureType.TRANSFORMED_VALUE)
				.setDataType(DataType.DOUBLE)
				.setOpType(OpType.CONTINUOUS)
				.setExpression(PMMLUtil.createApply("logit", new FieldRef(decisionFunction.getName())));

			categoryProbabilities.put(targetCategory, logitDecisionFunction.getName());

			Output output = new Output()
				.addOutputFields(decisionFunction, logitDecisionFunction);

			RegressionModel regressionModel = new RegressionModel(MiningFunctionType.REGRESSION, regressionMiningSchema, null)
				.addRegressionTables(regressionTable)
				.setOutput(output);

			Segment segment = new Segment()
				.setPredicate(new True())
				.setModel(regressionModel);

			segmentation.addSegments(segment);
		}

		if(numberOfClasses == 1){
			String targetCategory = String.valueOf(classes.get(0));

			MiningField miningField = PMMLUtil.createMiningField(Iterables.getOnlyElement(categoryProbabilities.values()));

			MiningSchema miningSchema = new MiningSchema()
				.addMiningFields(miningField);

			NumericPredictor numericPredictor = new NumericPredictor(miningField.getName(), -1d);

			RegressionTable regressionTable = RegressionModelUtil.encodeRegressionTable(numericPredictor, 1d);

			OutputField logitDecisionFunction = PMMLUtil.createPredictedField(FieldName.create("logitDecisionFunction_" + targetCategory));

			categoryProbabilities.put(targetCategory, logitDecisionFunction.getName());

			Output output = new Output()
				.addOutputFields(logitDecisionFunction);

			RegressionModel regressionModel = new RegressionModel(MiningFunctionType.REGRESSION, miningSchema, null)
				.addRegressionTables(regressionTable)
				.setOutput(output);

			Segment segment = new Segment()
				.setPredicate(new True())
				.setModel(regressionModel);

			segmentation.addSegments(segment);
		}

		MiningSchema classificationMiningSchema = new MiningSchema();

		classificationMiningSchema.addMiningFields(PMMLUtil.createMiningField(dataField.getName(), FieldUsageType.TARGET));

		RegressionModel regressionModel = new RegressionModel(MiningFunctionType.CLASSIFICATION, classificationMiningSchema, null)
			.setNormalizationMethod(numberOfClasses > 1 ? RegressionNormalizationMethodType.SIMPLEMAX : null);

		Collection<Map.Entry<String, FieldName>> entries = categoryProbabilities.entrySet();
		for(Map.Entry<String, FieldName> entry : entries){
			MiningField miningField = PMMLUtil.createMiningField(entry.getValue());

			classificationMiningSchema.addMiningFields(miningField);

			NumericPredictor numericPredictor = new NumericPredictor(miningField.getName(), 1d);

			RegressionTable regressionTable = RegressionModelUtil.encodeRegressionTable(numericPredictor, 0d)
				.setTargetCategory(entry.getKey());

			regressionModel.addRegressionTables(regressionTable);
		}

		Segment segment = new Segment()
			.setPredicate(new True())
			.setModel(regressionModel);

		segmentation.addSegments(segment);

		MiningSchema miningSchema = PMMLUtil.createMiningSchema(dataFields);

		Output output = new Output(PMMLUtil.createProbabilityFields(dataField));

		MiningModel miningModel = new MiningModel(MiningFunctionType.CLASSIFICATION, miningSchema)
			.setSegmentation(segmentation)
			.setOutput(output);

		return miningModel;
	}

	@Override
	public PMML encodePMML(){
		PMML pmml = super.encodePMML();

		TransformationDictionary transformationDictionary = new TransformationDictionary()
			.addDefineFunctions(encodeLogitFunction());

		pmml.setTransformationDictionary(transformationDictionary);

		return pmml;
	}

	public DefineFunction encodeLogitFunction(){
		FieldName name = FieldName.create("value");

		ParameterField parameterField = new ParameterField(name)
			.setDataType(DataType.DOUBLE)
			.setOpType(OpType.CONTINUOUS);

		// "1 / (1 + exp(-1 * $name))"
		Expression expression = PMMLUtil.createApply("/", PMMLUtil.createConstant(1d), PMMLUtil.createApply("+", PMMLUtil.createConstant(1d), PMMLUtil.createApply("exp", PMMLUtil.createApply("*", PMMLUtil.createConstant(-1d), new FieldRef(name)))));

		DefineFunction defineFunction = new DefineFunction("logit", OpType.CONTINUOUS, null)
			.setDataType(DataType.DOUBLE)
			.setOpType(OpType.CONTINUOUS)
			.addParameterFields(parameterField)
			.setExpression(expression);

		return defineFunction;
	}

	public List<? extends Number> getCoef(){
		return (List)ClassDictUtil.getArray(this, "coef_");
	}

	public List<? extends Number> getIntercept(){
		return (List)ClassDictUtil.getArray(this, "intercept_");
	}

	private int[] getCoefShape(){
		NDArrayWrapper arrayWrapper = (NDArrayWrapper)get("coef_");

		NDArray array = arrayWrapper.getContent();

		return NDArrayUtil.getShape(array);
	}

	static
	private <E> List<E> getRow(List<E> values, int rows, int columns, int row){

		if(values.size() != (rows * columns)){
			throw new IllegalArgumentException();
		}

		int offset = (row * columns);

		return values.subList(offset, offset + columns);
	}
}