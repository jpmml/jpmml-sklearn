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

import com.google.common.collect.Iterables;
import org.dmg.pmml.regression.RegressionModel;
import org.jpmml.converter.Schema;
import org.jpmml.converter.regression.RegressionModelUtil;
import sklearn.Regressor;

public class LinearRegressor extends Regressor {

	public LinearRegressor(String module, String name){
		super(module, name);
	}

	@Override
	public int getNumberOfFeatures(){
		int[] shape = getCoefShape();

		return shape[0];
	}

	@Override
	public RegressionModel encodeModel(Schema schema){
		List<? extends Number> coef = getCoef();
		List<? extends Number> intercept = getIntercept();

		return RegressionModelUtil.createRegression(schema.getFeatures(), coef, Iterables.getOnlyElement(intercept), null, schema);
	}

	public List<? extends Number> getCoef(){
		return getNumberArray("coef_");
	}

	public int[] getCoefShape(){
		return getArrayShape("coef_", 1);
	}

	public List<? extends Number> getIntercept(){
		return getNumberArray("intercept_");
	}
}