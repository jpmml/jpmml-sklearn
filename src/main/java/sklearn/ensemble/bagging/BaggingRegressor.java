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
package sklearn.ensemble.bagging;

import java.util.List;

import org.dmg.pmml.MiningFunctionType;
import org.dmg.pmml.MiningModel;
import org.jpmml.sklearn.Schema;
import sklearn.Regressor;

public class BaggingRegressor extends Regressor {

	public BaggingRegressor(String module, String name){
		super(module, name);
	}

	@Override
	public MiningModel encodeModel(Schema schema){
		List<Regressor> estimators = getEstimators();
		List<List<Integer>> estimatorsFeatures = getEstimatorsFeatures();

		MiningModel miningModel = BaggingUtil.encodeBagging(estimators, estimatorsFeatures, MiningFunctionType.REGRESSION, schema);

		return miningModel;
	}

	public List<Regressor> getEstimators(){
		List<?> estimators = (List)get("estimators_");

		return BaggingUtil.transformEstimators(estimators, Regressor.class);
	}

	public List<List<Integer>> getEstimatorsFeatures(){
		List<?> estimatorsFeatures = (List)get("estimators_features_");

		return BaggingUtil.transformEstimatorsFeatures(estimatorsFeatures);
	}
}