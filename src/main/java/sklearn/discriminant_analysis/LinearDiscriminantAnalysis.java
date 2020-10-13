/*
 * Copyright (c) 2019 Villu Ruusmann
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
package sklearn.discriminant_analysis;

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
import org.jpmml.sklearn.SkLearnUtil;
import sklearn.linear_model.LinearClassifier;

public class LinearDiscriminantAnalysis extends LinearClassifier {

	public LinearDiscriminantAnalysis(String module, String name){
		super(module, name);
	}

	@Override
	public Model encodeModel(Schema schema){
		int[] shape = getCoefShape();

		int numberOfClasses = shape[0];
		int numberOfFeatures = shape[1];

		if(numberOfClasses == 1){
			return encodeBinaryModel(schema);
		} else

		{
			return encodeMultinomialModel(schema);
		}
	}

	private Model encodeBinaryModel(Schema schema){
		return super.encodeModel(schema);
	}

	private Model encodeMultinomialModel(Schema schema){
		String sklearnVersion = getSkLearnVersion();
		int[] shape = getCoefShape();

		int numberOfClasses = shape[0];
		int numberOfFeatures = shape[1];

		List<? extends Number> coef = getCoef();
		List<? extends Number> intercept = getIntercept();

		CategoricalLabel categoricalLabel = (CategoricalLabel)schema.getLabel();
		List<? extends Feature> features = schema.getFeatures();

		// See https://github.com/scikit-learn/scikit-learn/issues/6848
		boolean corrected = (sklearnVersion != null && SkLearnUtil.compareVersion(sklearnVersion, "0.21") >= 0);

		if(!corrected){
			return super.encodeModel(schema);
		} // End if

		if(numberOfClasses >= 3){
			SchemaUtil.checkSize(numberOfClasses, categoricalLabel);

			Schema segmentSchema = (schema.toAnonymousRegressorSchema(DataType.DOUBLE)).toEmptySchema();

			List<RegressionModel> regressionModels = new ArrayList<>();

			for(int i = 0, rows = categoricalLabel.size(); i < rows; i++){
				RegressionModel regressionModel = RegressionModelUtil.createRegression(features, CMatrixUtil.getRow(coef, numberOfClasses, numberOfFeatures, i), intercept.get(i), RegressionModel.NormalizationMethod.NONE, segmentSchema)
					.setOutput(ModelUtil.createPredictedOutput(FieldNameUtil.create("decisionFunction", categoricalLabel.getValue(i)), OpType.CONTINUOUS, DataType.DOUBLE));

				regressionModels.add(regressionModel);
			}

			return MiningModelUtil.createClassification(regressionModels, RegressionModel.NormalizationMethod.SOFTMAX, true, schema);
		} else

		{
			throw new IllegalArgumentException();
		}
	}
}