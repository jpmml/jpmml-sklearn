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
import org.dmg.pmml.regression.RegressionModel;
import org.jpmml.converter.CMatrixUtil;
import org.jpmml.converter.CategoricalLabel;
import org.jpmml.converter.Feature;
import org.jpmml.converter.FieldNameUtil;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.Schema;
import org.jpmml.converter.SchemaUtil;
import org.jpmml.converter.mining.MiningModelUtil;
import org.jpmml.converter.regression.RegressionModelUtil;
import sklearn.Classifier;

public class LinearClassifier extends Classifier {

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

		List<? extends Number> coef = getCoef();
		List<? extends Number> intercept = getIntercept();

		CategoricalLabel categoricalLabel = (CategoricalLabel)schema.getLabel();
		List<? extends Feature> features = schema.getFeatures();

		if(numberOfClasses == 1){
			SchemaUtil.checkSize(2, categoricalLabel);

			return RegressionModelUtil.createBinaryLogisticClassification(features, CMatrixUtil.getRow(coef, numberOfClasses, numberOfFeatures, 0), intercept.get(0), RegressionModel.NormalizationMethod.LOGIT, hasProbabilityDistribution, schema);
		} else

		if(numberOfClasses >= 3){
			SchemaUtil.checkSize(numberOfClasses, categoricalLabel);

			Schema segmentSchema = (schema.toAnonymousRegressorSchema(DataType.DOUBLE)).toEmptySchema();

			List<RegressionModel> regressionModels = new ArrayList<>();

			for(int i = 0, rows = categoricalLabel.size(); i < rows; i++){
				RegressionModel regressionModel = RegressionModelUtil.createRegression(features, CMatrixUtil.getRow(coef, numberOfClasses, numberOfFeatures, i), intercept.get(i), RegressionModel.NormalizationMethod.LOGIT, segmentSchema)
					.setOutput(ModelUtil.createPredictedOutput(FieldNameUtil.create("decisionFunction", categoricalLabel.getValue(i)), OpType.CONTINUOUS, DataType.DOUBLE));

				regressionModels.add(regressionModel);
			}

			return MiningModelUtil.createClassification(regressionModels, RegressionModel.NormalizationMethod.SIMPLEMAX, hasProbabilityDistribution, schema);
		} else

		{
			throw new IllegalArgumentException();
		}
	}

	public List<? extends Number> getCoef(){
		return getNumberArray("coef_");
	}

	public int[] getCoefShape(){
		return getArrayShape("coef_", 2);
	}

	public List<? extends Number> getIntercept(){
		return getNumberArray("intercept_");
	}
}