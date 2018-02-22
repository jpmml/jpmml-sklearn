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
package sklearn.linear_model.logistic;

import java.util.ArrayList;
import java.util.List;

import org.dmg.pmml.DataType;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.Model;
import org.dmg.pmml.regression.RegressionModel;
import org.dmg.pmml.regression.RegressionTable;
import org.jpmml.converter.CMatrixUtil;
import org.jpmml.converter.CategoricalLabel;
import org.jpmml.converter.Feature;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.Schema;
import org.jpmml.converter.ValueUtil;
import org.jpmml.converter.regression.RegressionModelUtil;
import sklearn.ClassifierUtil;
import sklearn.linear_model.LinearClassifier;

public class LogisticRegression extends LinearClassifier {

	public LogisticRegression(String module, String name){
		super(module, name);
	}

	@Override
	public Model encodeModel(Schema schema){
		String multiClass = getMultiClass();

		if(("multinomial").equals(multiClass)){
			return encodeMultinomialModel(schema);
		} else

		if(("ovr").equals(multiClass)){
			return encodeOvRModel(schema);
		} else

		{
			throw new IllegalArgumentException(multiClass);
		}
	}

	private Model encodeMultinomialModel(Schema schema){
		int[] shape = getCoefShape();

		int numberOfClasses = shape[0];
		int numberOfFeatures = shape[1];

		List<? extends Number> coef = getCoef();
		List<? extends Number> intercepts = getIntercept();

		CategoricalLabel categoricalLabel = (CategoricalLabel)schema.getLabel();

		List<Feature> features = schema.getFeatures();

		if(numberOfClasses == 1){
			return encodeOvRModel(schema);
		} else

		if(numberOfClasses >= 3){
			ClassifierUtil.checkSize(numberOfClasses, categoricalLabel);

			List<RegressionTable> regressionTables = new ArrayList<>();

			for(int i = 0; i < categoricalLabel.size(); i++){
				RegressionTable regressionTable = RegressionModelUtil.createRegressionTable(features, ValueUtil.asDoubles(CMatrixUtil.getRow(coef, numberOfClasses, numberOfFeatures, i)), ValueUtil.asDouble(intercepts.get(i)))
					.setTargetCategory(categoricalLabel.getValue(i));

				regressionTables.add(regressionTable);
			}

			RegressionModel regressionModel = new RegressionModel(MiningFunction.CLASSIFICATION, ModelUtil.createMiningSchema(categoricalLabel), regressionTables)
				.setNormalizationMethod(RegressionModel.NormalizationMethod.SOFTMAX)
				.setOutput(ModelUtil.createProbabilityOutput(DataType.DOUBLE, categoricalLabel));

			return regressionModel;
		} else

		{
			throw new IllegalArgumentException();
		}
	}

	private Model encodeOvRModel(Schema schema){
		return super.encodeModel(schema);
	}

	public String getMultiClass(){
		return (String)get("multi_class");
	}
}