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

import com.google.common.collect.Iterables;
import org.dmg.pmml.Model;
import org.dmg.pmml.regression.RegressionModel;
import org.jpmml.converter.CMatrixUtil;
import org.jpmml.converter.Feature;
import org.jpmml.converter.Schema;
import org.jpmml.converter.regression.RegressionModelUtil;
import sklearn.SkLearnRegressor;

public class LinearRegressor extends SkLearnRegressor {

	public LinearRegressor(String module, String name){
		super(module, name);
	}

	@Override
	public int getNumberOfFeatures(){
		int[] shape = getCoefShape();

		if(shape.length == 2){
			return shape[1];
		}

		return shape[0];
	}

	@Override
	public int getNumberOfOutputs(){
		int[] shape = getCoefShape();

		if(shape.length == 2){
			return shape[0];
		}

		return 1;
	}

	@Override
	public Model encodeModel(Schema schema){
		List<Number> coef = getCoef();
		List<Number> intercept = getIntercept();

		List<? extends Feature> features = schema.getFeatures();

		RegressionModel.NormalizationMethod normalizationMethod = getNormalizationMethod();

		int numberOfOutputs = getNumberOfOutputs();
		if(numberOfOutputs == 1){
			return RegressionModelUtil.createRegression(features, coef, Iterables.getOnlyElement(intercept), normalizationMethod, schema);
		} else

		if(numberOfOutputs >= 2){
			List<List<? extends Number>> coefficients = new ArrayList<>();

			for(int i = 0, max = numberOfOutputs; i < max; i++){
				coefficients.add(CMatrixUtil.getRow(coef, numberOfOutputs, features.size(), i));
			}

			return RegressionModelUtil.createMultiRegression(features, coefficients, intercept, normalizationMethod, schema);
		} else

		{
			throw new IllegalArgumentException();
		}
	}

	protected RegressionModel.NormalizationMethod getNormalizationMethod(){
		return RegressionModel.NormalizationMethod.NONE;
	}

	public List<Number> getCoef(){
		return getNumberArray("coef_");
	}

	public int[] getCoefShape(){
		return getArrayShape("coef_");
	}

	public List<Number> getIntercept(){
		return getNumberArray("intercept_");
	}
}