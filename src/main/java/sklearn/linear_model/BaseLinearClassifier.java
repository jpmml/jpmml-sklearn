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
import org.dmg.pmml.OpType;
import org.dmg.pmml.mining.MiningModel;
import org.dmg.pmml.regression.RegressionModel;
import org.jpmml.converter.CMatrixUtil;
import org.jpmml.converter.CategoricalLabel;
import org.jpmml.converter.ContinuousLabel;
import org.jpmml.converter.FunctionTransformation;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.Schema;
import org.jpmml.converter.mining.MiningModelUtil;
import org.jpmml.sklearn.ClassDictUtil;
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
	public MiningModel encodeModel(Schema schema){
		int[] shape = getCoefShape();

		int numberOfClasses = shape[0];
		int numberOfFeatures = shape[1];

		boolean hasProbabilityDistribution = hasProbabilityDistribution();

		List<? extends Number> coefficients = getCoef();
		List<? extends Number> intercepts = getIntercept();

		Schema segmentSchema = new Schema(new ContinuousLabel(null, DataType.DOUBLE), schema.getFeatures());

		CategoricalLabel categoricalLabel = (CategoricalLabel)schema.getLabel();

		if(numberOfClasses == 1){
			EstimatorUtil.checkSize(2, categoricalLabel);

			RegressionModel regressionModel = BaseLinearUtil.encodeRegressionModel(intercepts.get(0), CMatrixUtil.getRow(coefficients, numberOfClasses, numberOfFeatures, 0), segmentSchema)
				.setOutput(ModelUtil.createPredictedOutput(FieldName.create("decisionFunction_" + categoricalLabel.getValue(1)), OpType.CONTINUOUS, DataType.DOUBLE));

			return MiningModelUtil.createBinaryLogisticClassification(schema, regressionModel, RegressionModel.NormalizationMethod.SOFTMAX, 0d, 1d, hasProbabilityDistribution);
		} else

		if(numberOfClasses >= 2){
			EstimatorUtil.checkSize(numberOfClasses, categoricalLabel);

			List<RegressionModel> regressionModels = new ArrayList<>();

			for(int i = 0, rows = categoricalLabel.size(); i < rows; i++){
				RegressionModel regressionModel = BaseLinearUtil.encodeRegressionModel(intercepts.get(i), CMatrixUtil.getRow(coefficients, numberOfClasses, numberOfFeatures, i), segmentSchema)
					.setOutput(ModelUtil.createPredictedOutput(FieldName.create("decisionFunction_" + categoricalLabel.getValue(i)), OpType.CONTINUOUS, DataType.DOUBLE, new FunctionTransformation("logit")));

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
}