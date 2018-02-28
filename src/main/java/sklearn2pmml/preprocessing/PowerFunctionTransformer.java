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
package sklearn2pmml.preprocessing;

import java.util.ArrayList;
import java.util.List;

import org.jpmml.converter.BinaryFeature;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.Feature;
import org.jpmml.converter.PowerFeature;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn.Transformer;

public class PowerFunctionTransformer extends Transformer {

	public PowerFunctionTransformer(String module, String name){
		super(module, name);
	}

	@Override
	public List<Feature> encodeFeatures(List<Feature> features, SkLearnEncoder encoder){
		Integer power = getPower();

		List<Feature> result = new ArrayList<>();

		for(Feature feature : features){

			if(feature instanceof BinaryFeature){
				BinaryFeature binaryFeature = (BinaryFeature)feature;

				result.add(binaryFeature);
			} else

			{
				ContinuousFeature continuousFeature = feature.toContinuousFeature();

				result.add(new PowerFeature(encoder, continuousFeature.getName(), continuousFeature.getDataType(), power));
			}
		}

		return result;
	}

	public Integer getPower(){
		return (Integer)get("power");
	}
}