/*
 * Copyright (c) 2021 Villu Ruusmann
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
package sklearn.ensemble.hist_gradient_boosting;

import java.util.List;

import com.google.common.base.Function;
import com.google.common.collect.Lists;
import h2o.estimators.BaseEstimator;
import org.jpmml.python.HasArray;

public class BinMapper extends BaseEstimator {

	public BinMapper(String module, String name){
		super(module, name);
	}

	public int getNumberOfBins(){
		return getInteger("n_bins");
	}

	public List<List<Number>> getBinThresholds(){
		List<? extends HasArray> arrays = getList("bin_thresholds_", HasArray.class);

		Function<HasArray, List<Number>> function = new Function<HasArray, List<Number>>(){

			@Override
			public List<Number> apply(HasArray hasArray){
				return (List)hasArray.getArrayContent();
			}
		};

		return Lists.transform(arrays, function);
	}
}