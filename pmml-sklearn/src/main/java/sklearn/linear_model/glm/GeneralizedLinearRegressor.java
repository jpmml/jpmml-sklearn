/*
 * Copyright (c) 2020 Villu Ruusmann
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
package sklearn.linear_model.glm;

import java.util.List;

import org.dmg.pmml.regression.RegressionModel;
import org.jpmml.converter.Schema;
import org.jpmml.converter.regression.RegressionModelUtil;
import sklearn.linear_model.LinearRegressor;

public class GeneralizedLinearRegressor extends LinearRegressor {

	public GeneralizedLinearRegressor(String module, String name){
		super(module, name);
	}

	@Override
	protected RegressionModel createRegression(List<? extends Number> coef, Number intercept, Schema schema){
		return RegressionModelUtil.createRegression(schema.getFeatures(), coef, intercept, RegressionModel.NormalizationMethod.EXP, schema);
	}
}