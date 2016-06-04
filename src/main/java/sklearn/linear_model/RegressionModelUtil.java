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

import org.dmg.pmml.CategoricalPredictor;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.MiningFunctionType;
import org.dmg.pmml.MiningSchema;
import org.dmg.pmml.NumericPredictor;
import org.dmg.pmml.RegressionModel;
import org.dmg.pmml.RegressionTable;
import org.jpmml.converter.BinaryFeature;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.Feature;
import org.jpmml.converter.FeatureSchema;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.ValueUtil;
import org.jpmml.sklearn.LoggerUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class RegressionModelUtil {

	private RegressionModelUtil(){
	}

	static
	public RegressionModel encodeRegressionModel(List<? extends Number> coefficients, Number intercept, FeatureSchema schema){
		RegressionTable regressionTable = encodeRegressionTable(coefficients, intercept, schema);

		MiningSchema miningSchema = ModelUtil.createMiningSchema(schema, regressionTable);

		RegressionModel regressionModel = new RegressionModel(MiningFunctionType.REGRESSION, miningSchema, null)
			.addRegressionTables(regressionTable);

		return regressionModel;
	}

	static
	public RegressionTable encodeRegressionTable(List<? extends Number> coefficients, Number intercept, FeatureSchema schema){
		List<Feature> features = schema.getFeatures();

		if(coefficients.size() != features.size()){
			throw new IllegalArgumentException();
		}

		RegressionTable regressionTable = new RegressionTable(ValueUtil.asDouble(intercept));

		List<FieldName> unusedNames = new ArrayList<>();

		for(int i = 0; i < coefficients.size(); i++){
			Number coefficient = coefficients.get(i);
			Feature feature = features.get(i);

			if(ValueUtil.isZero(coefficient)){
				unusedNames.add(feature.getName());

				continue;
			} // End if

			if(feature instanceof ContinuousFeature){
				ContinuousFeature continuousFeature = (ContinuousFeature)feature;

				NumericPredictor numericPredictor = new NumericPredictor(continuousFeature.getName(), ValueUtil.asDouble(coefficient));

				regressionTable.addNumericPredictors(numericPredictor);
			} else

			if(feature instanceof BinaryFeature){
				BinaryFeature binaryFeature = (BinaryFeature)feature;

				CategoricalPredictor categoricalPredictor = new CategoricalPredictor(binaryFeature.getName(), binaryFeature.getValue(), ValueUtil.asDouble(coefficient));

				regressionTable.addCategoricalPredictors(categoricalPredictor);
			} else

			{
				throw new IllegalArgumentException();
			}
		}

		if(!unusedNames.isEmpty()){
			logger.info("Skipped {} active field(s): {}", unusedNames.size(), LoggerUtil.formatNameList(unusedNames));
		}

		return regressionTable;
	}

	static
	public RegressionTable encodeRegressionTable(NumericPredictor numericPredictor, Number intercept){
		RegressionTable regressionTable = new RegressionTable(ValueUtil.asDouble(intercept))
			.addNumericPredictors(numericPredictor);

		return regressionTable;
	}

	private static final Logger logger = LoggerFactory.getLogger(RegressionModelUtil.class);
}