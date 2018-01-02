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
import org.jpmml.converter.ValueUtil;
import org.jpmml.converter.regression.RegressionModelUtil;
import sklearn.Regressor;

abstract
public class BaseLinearRegressor extends Regressor {

	public BaseLinearRegressor(String module, String name){
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

		return RegressionModelUtil.createRegression(schema.getFeatures(), ValueUtil.asDoubles(coef), ValueUtil.asDouble(Iterables.getOnlyElement(intercept)), null, schema);
	}

	public List<? extends Number> getCoef(){
		return getArray("coef_", Number.class);
	}

	public int[] getCoefShape(){
		return getArrayShape("coef_", 1);
	}

	public List<? extends Number> getIntercept(){
		return getArray("intercept_", Number.class);
	}
}