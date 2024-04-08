/*
 * Copyright (c) 2019 Villu Ruusmann
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
package sklearn2pmml.ensemble;

import java.util.List;

import com.google.common.collect.Iterables;
import org.dmg.pmml.Model;
import org.jpmml.converter.Schema;
import sklearn.Regressor;
import sklearn.linear_model.LinearRegressor;
import sklearn.preprocessing.MultiOneHotEncoder;

public class GBDTLMRegressor extends Regressor {

	public GBDTLMRegressor(String module, String name){
		super(module, name);
	}

	@Override
	public Model encodeModel(Schema schema){
		Regressor gbdt = getGBDT();
		MultiOneHotEncoder ohe = getOHE();
		LinearRegressor lm = getLM();

		List<Number> coef = lm.getCoef();
		List<Number> intercept = lm.getIntercept();

		return GBDTUtil.encodeModel(gbdt, ohe, coef, Iterables.getOnlyElement(intercept), schema);
	}

	public Regressor getGBDT(){
		return get("gbdt_", Regressor.class);
	}

	public LinearRegressor getLM(){
		return get("lm_", LinearRegressor.class);
	}

	public MultiOneHotEncoder getOHE(){
		return get("ohe_", MultiOneHotEncoder.class);
	}
}