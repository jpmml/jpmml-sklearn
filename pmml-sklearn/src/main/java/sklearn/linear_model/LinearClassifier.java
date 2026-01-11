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

import org.dmg.pmml.DataType;
import org.dmg.pmml.Model;
import org.dmg.pmml.OpType;
import org.dmg.pmml.mining.MiningModel;
import org.dmg.pmml.regression.RegressionModel;
import org.jpmml.converter.CMatrixUtil;
import org.jpmml.converter.CategoricalLabel;
import org.jpmml.converter.Feature;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.Schema;
import org.jpmml.converter.mining.MiningModelUtil;
import org.jpmml.converter.regression.RegressionModelUtil;
import sklearn.HasMultiDecisionFunctionField;
import sklearn.SkLearnClassifier;

public class LinearClassifier extends SkLearnClassifier implements HasMultiDecisionFunctionField {

	public LinearClassifier(String module, String name){
		super(module, name);
	}

	@Override
	public int getNumberOfFeatures(){
		int[] shape = getCoefShape();

		return shape[1];
	}

	@Override
	public Model encodeModel(Schema schema){
		int[] shape = getCoefShape();

		int numberOfClasses = shape[0];
		int numberOfFeatures = shape[1];

		boolean hasProbabilityDistribution = hasProbabilityDistribution();

		List<Number> coef = getCoef();
		List<Number> intercept = getIntercept();

		CategoricalLabel categoricalLabel = schema.requireCategoricalLabel();
		List<? extends Feature> features = schema.getFeatures();

		if(numberOfClasses == 1){
			categoricalLabel.expectCardinality(2);

			RegressionModel regressionModel = RegressionModelUtil.createBinaryLogisticClassification(features, CMatrixUtil.getRow(coef, numberOfClasses, numberOfFeatures, 0), intercept.get(0), RegressionModel.NormalizationMethod.LOGIT, false, schema);

			if(hasProbabilityDistribution){
				encodePredictProbaOutput(regressionModel, DataType.DOUBLE, categoricalLabel);
			}

			return regressionModel;
		} else

		if(numberOfClasses >= 3){
			categoricalLabel.expectCardinality(numberOfClasses);

			Schema segmentSchema = (schema.toAnonymousRegressorSchema(DataType.DOUBLE)).toEmptySchema();

			List<Model> models = new ArrayList<>();

			for(int i = 0, rows = categoricalLabel.size(); i < rows; i++){
				Model model = RegressionModelUtil.createRegression(features, CMatrixUtil.getRow(coef, numberOfClasses, numberOfFeatures, i), intercept.get(i), RegressionModel.NormalizationMethod.LOGIT, segmentSchema)
					.setOutput(ModelUtil.createPredictedOutput(getMultiDecisionFunctionField(categoricalLabel.getValue(i)), OpType.CONTINUOUS, DataType.DOUBLE));

				models.add(model);
			}

			MiningModel miningModel = MiningModelUtil.createClassification(models, RegressionModel.NormalizationMethod.SIMPLEMAX, false, schema);

			if(hasProbabilityDistribution){
				encodePredictProbaOutput(miningModel, DataType.DOUBLE, categoricalLabel);
			}

			return miningModel;
		} else

		{
			throw new IllegalArgumentException();
		}
	}

	public List<Number> getCoef(){
		return getNumberArray("coef_");
	}

	public int[] getCoefShape(){
		return getArrayShape("coef_", 2);
	}

	public List<Number> getIntercept(){
		return getNumberArray("intercept_");
	}
}