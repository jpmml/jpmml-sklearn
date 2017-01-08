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
import java.util.Collections;
import java.util.List;
import java.util.Set;

import org.dmg.pmml.DataType;
import org.dmg.pmml.DefineFunction;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.FieldRef;
import org.dmg.pmml.OpType;
import org.dmg.pmml.Output;
import org.dmg.pmml.OutputField;
import org.dmg.pmml.ResultFeature;
import org.dmg.pmml.mining.MiningModel;
import org.dmg.pmml.regression.RegressionModel;
import org.jpmml.converter.CategoricalLabel;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.converter.Schema;
import org.jpmml.converter.mining.MiningModelUtil;
import org.jpmml.sklearn.ClassDictUtil;
import org.jpmml.sklearn.MatrixUtil;
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

		return shape[1];
	}

	@Override
	public boolean requiresContinuousInput(){
		return false;
	}

	@Override
	public MiningModel encodeModel(Schema schema){
		int[] shape = getCoefShape();

		int numberOfClasses = shape[0];
		int numberOfFeatures = shape[1];

		boolean hasProbabilityDistribution = hasProbabilityDistribution();

		List<? extends Number> coefficients = getCoef();
		List<? extends Number> intercepts = getIntercept();

		Schema segmentSchema = schema.toAnonymousSchema();

		CategoricalLabel categoricalLabel = (CategoricalLabel)segmentSchema.getLabel();

		if(numberOfClasses == 1){
			EstimatorUtil.checkSize(2, categoricalLabel);

			RegressionModel regressionModel = encodeCategoryRegressor(categoricalLabel.getValue(1), MatrixUtil.getRow(coefficients, numberOfClasses, numberOfFeatures, 0), intercepts.get(0), null, segmentSchema);

			return MiningModelUtil.createBinaryLogisticClassification(schema, regressionModel, -1d, hasProbabilityDistribution);
		} else

		if(numberOfClasses >= 2){
			EstimatorUtil.checkSize(numberOfClasses, categoricalLabel);

			List<RegressionModel> regressionModels = new ArrayList<>();

			for(int i = 0, rows = categoricalLabel.size(); i < rows; i++){
				RegressionModel regressionModel = encodeCategoryRegressor(categoricalLabel.getValue(i), MatrixUtil.getRow(coefficients, numberOfClasses, numberOfFeatures, i), intercepts.get(i), "logit", segmentSchema);

				regressionModels.add(regressionModel);
			}

			return MiningModelUtil.createClassification(schema, regressionModels, RegressionModel.NormalizationMethod.SIMPLEMAX, hasProbabilityDistribution);
		} else

		{
			throw new IllegalArgumentException();
		}
	}

	@Override
	public Set<DefineFunction> encodeDefineFunctions(){
		return Collections.singleton(EstimatorUtil.encodeLogitFunction());
	}

	public List<? extends Number> getCoef(){
		return (List)ClassDictUtil.getArray(this, "coef_");
	}

	public List<? extends Number> getIntercept(){
		return (List)ClassDictUtil.getArray(this, "intercept_");
	}

	private int[] getCoefShape(){
		return ClassDictUtil.getShape(this, "coef_", 2);
	}

	static
	private RegressionModel encodeCategoryRegressor(String targetCategory, List<? extends Number> coefficients, Number intercept, String outputTransformation, Schema schema){
		OutputField decisionFunction = new OutputField(FieldName.create("decisionFunction_" + targetCategory), DataType.DOUBLE)
			.setOpType(OpType.CONTINUOUS)
			.setResultFeature(ResultFeature.PREDICTED_VALUE)
			.setFinalResult(false);

		Output output = new Output()
			.addOutputFields(decisionFunction);

		if(outputTransformation != null){
			OutputField transformedDecisionFunction = new OutputField(FieldName.create(outputTransformation + "DecisionFunction_" + targetCategory), DataType.DOUBLE)
				.setOpType(OpType.CONTINUOUS)
				.setResultFeature(ResultFeature.TRANSFORMED_VALUE)
				.setFinalResult(false)
				.setExpression(PMMLUtil.createApply(outputTransformation, new FieldRef(decisionFunction.getName())));

			output.addOutputFields(transformedDecisionFunction);
		}

		RegressionModel regressionModel = BaseLinearUtil.encodeRegressionModel(intercept, coefficients, schema)
			.setOutput(output);

		return regressionModel;
	}
}