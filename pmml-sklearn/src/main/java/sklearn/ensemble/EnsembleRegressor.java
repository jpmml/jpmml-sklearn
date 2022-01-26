/*
 * Copyright (c) 2016 Villu Ruusmann
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
package sklearn.ensemble;

import java.util.List;

import org.dmg.pmml.DataType;
import org.dmg.pmml.OpType;
import sklearn.HasEstimatorEnsemble;
import sklearn.Regressor;

abstract
public class EnsembleRegressor extends Regressor implements HasEstimatorEnsemble<Regressor> {

	public EnsembleRegressor(String module, String name){
		super(module, name);
	}

	@Override
	public OpType getOpType(){
		Regressor baseEstimator = getBaseEstimator();

		return baseEstimator.getOpType();
	}

	@Override
	public DataType getDataType(){
		Regressor baseEstimator = getBaseEstimator();

		return baseEstimator.getDataType();
	}

	public Regressor getBaseEstimator(){
		return get("base_estimator_", Regressor.class);
	}

	@Override
	public List<? extends Regressor> getEstimators(){
		return getList("estimators_", Regressor.class);
	}
}