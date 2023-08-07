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
package sklearn.multioutput;

import java.util.List;

import org.dmg.pmml.Model;
import org.jpmml.converter.Schema;
import sklearn.Regressor;
import sklearn.SkLearnRegressor;

public class RegressorChain extends SkLearnRegressor {

	public RegressorChain(String module, String name){
		super(module, name);
	}

	@Override
	public int getNumberOfOutputs(){
		List<? extends Regressor> estimators = getEstimators();

		return estimators.size();
	}

	@Override
	public Model encodeModel(Schema schema){
		List<? extends Regressor> estimators = getEstimators();
		List<Integer> order = getOrder();

		return ChainUtil.encodeChain(estimators, order, schema);
	}

	public List<? extends Regressor> getEstimators(){
		return getList("estimators_", Regressor.class);
	}

	public List<Integer> getOrder(){
		return getIntegerArray("order_");
	}
}