/*
 * Copyright (c) 2017 Villu Ruusmann
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
package sklearn.dummy;

import java.util.Collections;
import java.util.List;

import com.google.common.collect.Iterables;
import org.dmg.pmml.regression.RegressionModel;
import org.jpmml.converter.Feature;
import org.jpmml.converter.Schema;
import org.jpmml.converter.regression.RegressionModelUtil;
import org.jpmml.sklearn.ClassDictUtil;
import sklearn.Regressor;

public class DummyRegressor extends Regressor {

	public DummyRegressor(String module, String name){
		super(module, name);
	}

	@Override
	public int getNumberOfFeatures(){
		return -1;
	}

	@Override
	public RegressionModel encodeModel(Schema schema){
		List<? extends Number> constant = getConstant();

		Number intercept = Iterables.getOnlyElement(constant);

		return RegressionModelUtil.createRegression(Collections.<Feature>emptyList(), Collections.<Double>emptyList(), intercept.doubleValue(), null, schema);
	}

	public List<? extends Number> getConstant(){
		return (List)ClassDictUtil.getArray(this, "constant_");
	}
}