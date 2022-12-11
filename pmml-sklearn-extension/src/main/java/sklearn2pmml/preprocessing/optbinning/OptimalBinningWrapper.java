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
package sklearn2pmml.preprocessing.optbinning;

import java.util.List;

import optbinning.OptimalBinning;
import org.jpmml.converter.Feature;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn.Transformer;

public class OptimalBinningWrapper extends Transformer {

	public OptimalBinningWrapper(String module, String name){
		super(module, name);
	}

	@Override
	public List<Feature> encodeFeatures(List<Feature> features, SkLearnEncoder encoder){
		String metric = getMetric();
		OptimalBinning optimalBinning = getOptimalBinning();

		if(!optimalBinning.containsKey("metric")){
			optimalBinning.setMetric(metric);
		}

		return optimalBinning.encodeFeatures(features, encoder);
	}

	public String getMetric(){
		return getString("metric");
	}

	public OptimalBinning getOptimalBinning(){
		return get("optimal_binning_", OptimalBinning.class);
	}
}