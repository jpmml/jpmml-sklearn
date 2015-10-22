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

import java.util.Iterator;
import java.util.List;

import org.dmg.pmml.DataField;
import org.dmg.pmml.NumericPredictor;
import org.dmg.pmml.RegressionTable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class RegressionModelUtil {

	private RegressionModelUtil(){
	}

	static
	public RegressionTable encodeRegressionTable(List<? extends Number> coefficients, Number intercept, List<DataField> dataFields){
		RegressionTable regressionTable = new RegressionTable(intercept.doubleValue());

		Iterator<? extends Number> it = coefficients.iterator();

		for(int i = 1; i < dataFields.size(); i++){
			DataField dataField = dataFields.get(i);

			Number coefficient = it.next();

			if(Double.compare(coefficient.doubleValue(), 0d) == 0){
				logger.info("Skipping predictor variable {} due to a zero coefficient", dataField.getName());

				continue;
			}

			NumericPredictor numericPredictor = new NumericPredictor(dataField.getName(), coefficient.doubleValue());

			regressionTable.addNumericPredictors(numericPredictor);
		}

		if(it.hasNext()){
			throw new IllegalArgumentException();
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