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
import org.dmg.pmml.DataField;
import org.dmg.pmml.RegressionModel;
import org.jpmml.sklearn.ClassDictUtil;
import sklearn.Regressor;

public class LinearRegression extends Regressor {

	public LinearRegression(String module, String name){
		super(module, name);
	}

	@Override
	public int getNumberOfFeatures(){
		return (Integer)get("rank_");
	}

	@Override
	public RegressionModel encodeModel(List<DataField> dataFields){
		return RegressionModelUtil.encodeRegressionModel(getCoef(), Iterables.getOnlyElement(getIntercept()), dataFields, true);
	}

	public List<? extends Number> getCoef(){
		return (List)ClassDictUtil.getArray(this, "coef_");
	}

	public List<? extends Number> getIntercept(){
		return (List)ClassDictUtil.getArray(this, "intercept_");
	}
}