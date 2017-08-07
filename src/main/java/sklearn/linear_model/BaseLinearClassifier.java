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
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.regression.RegressionModel;
import org.dmg.pmml.regression.RegressionTable;
import org.jpmml.converter.CMatrixUtil;
import org.jpmml.converter.CategoricalLabel;
import org.jpmml.converter.Feature;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.Schema;
import org.jpmml.converter.ValueUtil;
import org.jpmml.converter.regression.RegressionModelUtil;
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
	public RegressionModel encodeModel(Schema schema){
		int[] shape = getCoefShape();

		int numberOfClasses = shape[0];
		int numberOfFeatures = shape[1];

		boolean hasProbabilityDistribution = hasProbabilityDistribution();

		List<? extends Number> coef = getCoef();
		List<? extends Number> intercepts = getIntercept();

		CategoricalLabel categoricalLabel = (CategoricalLabel)schema.getLabel();

		List<Feature> features = schema.getFeatures();

		if(numberOfClasses == 1){
			EstimatorUtil.checkSize(2, categoricalLabel);

			return RegressionModelUtil.createBinaryLogisticClassification(features, ValueUtil.asDoubles(CMatrixUtil.getRow(coef, numberOfClasses, numberOfFeatures, 0)), ValueUtil.asDouble(intercepts.get(0)), RegressionModel.NormalizationMethod.LOGIT, hasProbabilityDistribution, schema);
		} else

		if(numberOfClasses >= 3){
			EstimatorUtil.checkSize(numberOfClasses, categoricalLabel);

			List<RegressionTable> regressionTables = new ArrayList<>();

			for(int i = 0, rows = categoricalLabel.size(); i < rows; i++){
				RegressionTable regressionTable = RegressionModelUtil.createRegressionTable(features, ValueUtil.asDoubles(CMatrixUtil.getRow(coef, numberOfClasses, numberOfFeatures, i)), ValueUtil.asDouble(intercepts.get(i)))
					.setTargetCategory(categoricalLabel.getValue(i));

				regressionTables.add(regressionTable);
			}

			RegressionModel regressionModel = new RegressionModel(MiningFunction.CLASSIFICATION, ModelUtil.createMiningSchema(categoricalLabel), regressionTables)
				.setNormalizationMethod(RegressionModel.NormalizationMethod.LOGIT)
				.setOutput(hasProbabilityDistribution ? ModelUtil.createProbabilityOutput(DataType.DOUBLE, categoricalLabel) : null);

			return regressionModel;
		} else

		{
			throw new IllegalArgumentException();
		}
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