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

import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.mining.MiningModel;
import org.dmg.pmml.mining.Segmentation;
import org.jpmml.converter.Schema;
import sklearn.Regressor;
import sklearn.ensemble.EnsembleRegressor;

public class BaggingRegressor extends EnsembleRegressor {

	public BaggingRegressor(String module, String name){
		super(module, name);
	}

	@Override
	public MiningModel encodeModel(Schema schema){
		List<Regressor> estimators = getEstimators();
		List<List<Number>> estimatorsFeatures = getEstimatorsFeatures();

		MiningModel miningModel = BaggingUtil.encodeBagging(estimators, estimatorsFeatures, MiningFunction.REGRESSION, Segmentation.MultipleModelMethod.AVERAGE, schema);

		return miningModel;
	}

	public List<List<Number>> getEstimatorsFeatures(){
		return getArrayList("estimators_features_", Number.class);
	}
}