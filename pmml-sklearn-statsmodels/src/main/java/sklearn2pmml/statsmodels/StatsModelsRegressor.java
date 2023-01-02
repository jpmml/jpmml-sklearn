/*
 * Copyright (c) 2022 Villu Ruusmann
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
package sklearn2pmml.statsmodels;

import org.dmg.pmml.Model;
import org.jpmml.converter.Schema;
import sklearn.Regressor;
import statsmodels.ResultsWrapper;

public class StatsModelsRegressor extends Regressor {

	public StatsModelsRegressor(String module, String name){
		super(module, name);
	}

	@Override
	public Model encodeModel(Schema schema){
		Boolean fitIntercept = getFitIntercept();
		ResultsWrapper results = getResults();

		if(fitIntercept){
			schema = StatsModelsUtil.addConstant(schema);
		}

		return results.encodeModel(schema);
	}

	public Boolean getFitIntercept(){
		return getBoolean("fit_intercept");
	}

	public ResultsWrapper getResults(){
		return get("results_", ResultsWrapper.class);
	}

	static {
		StatsModelsUtil.initOnce();
	}
}