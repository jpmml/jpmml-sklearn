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

import java.util.List;

import org.dmg.pmml.DataField;
import org.dmg.pmml.MiningFunctionType;
import org.dmg.pmml.MiningSchema;
import org.dmg.pmml.NumericPredictor;
import org.dmg.pmml.RegressionModel;
import org.dmg.pmml.RegressionTable;
import org.jpmml.converter.FieldCollector;
import org.jpmml.converter.PMMLUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class RegressionModelUtil {

	private RegressionModelUtil(){
	}

	static
	public RegressionModel encodeRegressionModel(List<? extends Number> coefficients, Number intercept, List<DataField> dataFields, boolean standalone){
		RegressionTable regressionTable = encodeRegressionTable(coefficients, intercept, dataFields);

		DataField dataField = dataFields.get(0);

		FieldCollector fieldCollector = new RegressionModelFieldCollector();
		fieldCollector.applyTo(regressionTable);

		MiningSchema miningSchema = PMMLUtil.createMiningSchema((standalone ? dataField : null), fieldCollector);

		RegressionModel regressionModel = new RegressionModel(MiningFunctionType.REGRESSION, miningSchema, null)
			.addRegressionTables(regressionTable);

		return regressionModel;
	}

	static
	public RegressionTable encodeRegressionTable(List<? extends Number> coefficients, Number intercept, List<DataField> dataFields){

		if(coefficients.size() != (dataFields.size() - 1)){
			throw new IllegalArgumentException();
		}

		RegressionTable regressionTable = new RegressionTable(intercept.doubleValue());

		for(int i = 0; i < coefficients.size(); i++){
			Number coefficient = coefficients.get(i);
			DataField dataField = dataFields.get(i + 1);

			if(Double.compare(coefficient.doubleValue(), 0d) == 0){
				logger.info("Skipping predictor variable {} due to a zero coefficient", dataField.getName());

				continue;
			}

			NumericPredictor numericPredictor = new NumericPredictor(dataField.getName(), coefficient.doubleValue());

			regressionTable.addNumericPredictors(numericPredictor);
		}

		return regressionTable;
	}

	static
	public RegressionTable encodeRegressionTable(NumericPredictor numericPredictor, Number intercept){
		RegressionTable regressionTable = new RegressionTable(intercept.doubleValue())
			.addNumericPredictors(numericPredictor);

		return regressionTable;
	}

	private static final Logger logger = LoggerFactory.getLogger(RegressionModelUtil.class);
}