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
package sklearn;

import java.util.ArrayList;
import java.util.List;

import org.dmg.pmml.DataType;
import org.dmg.pmml.DefineFunction;
import org.dmg.pmml.Expression;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.FieldRef;
import org.dmg.pmml.FieldUsageType;
import org.dmg.pmml.MiningField;
import org.dmg.pmml.MiningFunctionType;
import org.dmg.pmml.MiningModel;
import org.dmg.pmml.MiningSchema;
import org.dmg.pmml.Model;
import org.dmg.pmml.MultipleModelMethodType;
import org.dmg.pmml.NumericPredictor;
import org.dmg.pmml.OpType;
import org.dmg.pmml.Output;
import org.dmg.pmml.OutputField;
import org.dmg.pmml.ParameterField;
import org.dmg.pmml.RegressionModel;
import org.dmg.pmml.RegressionNormalizationMethodType;
import org.dmg.pmml.RegressionTable;
import org.dmg.pmml.Segmentation;
import org.jpmml.converter.MiningModelUtil;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.sklearn.Schema;
import sklearn.linear_model.RegressionModelUtil;

public class EstimatorUtil {

	private EstimatorUtil(){
	}

	static
	public Schema createSegmentSchema(Schema schema){
		Schema result = new Schema(null, schema.getTargetCategories(), schema.getActiveFields());

		return result;
	}

	static
	public MiningModel encodeBinomialClassifier(List<String> targetCategories, List<FieldName> probabilityFields, Model model, boolean hasProbabilityDistribution, Schema schema){

		if((targetCategories.size() != 2) || (targetCategories.size() != probabilityFields.size())){
			throw new IllegalArgumentException();
		}

		List<Model> models = new ArrayList<>();
		models.add(model);

		{
			MiningField miningField = ModelUtil.createMiningField(probabilityFields.get(0));

			NumericPredictor numericPredictor = new NumericPredictor(miningField.getName(), -1d);

			RegressionTable regressionTable = RegressionModelUtil.encodeRegressionTable(numericPredictor, 1d);

			OutputField outputField = ModelUtil.createPredictedField(probabilityFields.get(1));

			Output output = new Output()
				.addOutputFields(outputField);

			MiningSchema miningSchema = new MiningSchema()
				.addMiningFields(miningField);

			RegressionModel regressionModel = new RegressionModel(MiningFunctionType.REGRESSION, miningSchema, null)
				.addRegressionTables(regressionTable)
				.setOutput(output);

			models.add(regressionModel);
		}

		return encodeClassifier(targetCategories, probabilityFields, models, null, hasProbabilityDistribution, schema);
	}

	static
	public MiningModel encodeMultinomialClassifier(List<String> targetCategories, List<FieldName> probabilityFields, List<? extends Model> models, boolean hasProbabilityDistribution, Schema schema){
		return encodeClassifier(targetCategories, probabilityFields, models, RegressionNormalizationMethodType.SIMPLEMAX, hasProbabilityDistribution, schema);
	}

	static
	private MiningModel encodeClassifier(List<String> targetCategories, List<FieldName> probabilityFields, List<? extends Model> models, RegressionNormalizationMethodType normalizationMethod, boolean hasProbabilityDistribution, Schema schema){

		if(targetCategories.size() != probabilityFields.size()){
			throw new IllegalArgumentException();
		}

		List<Model> segmentationModels = new ArrayList<>(models);

		FieldName targetField = schema.getTargetField();

		{
			MiningSchema miningSchema = new MiningSchema();

			if(targetField != null){
				miningSchema.addMiningFields(ModelUtil.createMiningField(targetField, FieldUsageType.TARGET));
			}

			RegressionModel regressionModel = new RegressionModel(MiningFunctionType.CLASSIFICATION, miningSchema, null)
				.setNormalizationMethod(normalizationMethod);

			for(int i = 0; i < targetCategories.size(); i++){
				MiningField miningField = ModelUtil.createMiningField(probabilityFields.get(i));

				miningSchema.addMiningFields(miningField);

				NumericPredictor numericPredictor = new NumericPredictor(miningField.getName(), 1d);

				RegressionTable regressionTable = RegressionModelUtil.encodeRegressionTable(numericPredictor, 0d)
					.setTargetCategory(targetCategories.get(i));

				regressionModel.addRegressionTables(regressionTable);
			}

			segmentationModels.add(regressionModel);
		}

		MiningSchema miningSchema = ModelUtil.createMiningSchema(targetField, schema.getActiveFields());

		Segmentation segmentation = MiningModelUtil.createSegmentation(MultipleModelMethodType.MODEL_CHAIN, segmentationModels);

		Output output = (hasProbabilityDistribution ? encodeClassifierOutput(schema) : null);

		MiningModel miningModel = new MiningModel(MiningFunctionType.CLASSIFICATION, miningSchema)
			.setSegmentation(segmentation)
			.setOutput(output);

		return miningModel;
	}

	static
	public Output encodeClassifierOutput(Schema schema){
		List<String> targetCategories = schema.getTargetCategories();

		if(targetCategories == null || targetCategories.isEmpty()){
			return null;
		}

		Output output = new Output(ModelUtil.createProbabilityFields(targetCategories));

		return output;
	}

	static
	public DefineFunction encodeLogitFunction(){
		return encodeLossFunction("logit", -1d);
	}

	static
	public DefineFunction encodeAdaBoostFunction(){
		return encodeLossFunction("adaboost", -2d);
	}

	static
	private DefineFunction encodeLossFunction(String function, Number multiplier){
		FieldName name = FieldName.create("value");

		ParameterField parameterField = new ParameterField(name)
			.setDataType(DataType.DOUBLE)
			.setOpType(OpType.CONTINUOUS);

		// "1 / (1 + exp($multiplier * $name))"
		Expression expression = PMMLUtil.createApply("/", PMMLUtil.createConstant(1d), PMMLUtil.createApply("+", PMMLUtil.createConstant(1d), PMMLUtil.createApply("exp", PMMLUtil.createApply("*", PMMLUtil.createConstant(multiplier), new FieldRef(name)))));

		DefineFunction defineFunction = new DefineFunction(function, OpType.CONTINUOUS, null)
			.setDataType(DataType.DOUBLE)
			.setOpType(OpType.CONTINUOUS)
			.addParameterFields(parameterField)
			.setExpression(expression);

		return defineFunction;
	}
}