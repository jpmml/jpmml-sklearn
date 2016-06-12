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
package org.jpmml.sklearn;

import java.util.ArrayList;
import java.util.List;

import com.google.common.base.Function;
import com.google.common.collect.Lists;
import org.jpmml.converter.BinaryFeature;
import org.jpmml.converter.Feature;

public class LoggerUtil {

	private LoggerUtil(){
	}

	static
	public String formatNameList(List<Feature> features){
		Function<Feature, String> function = new Function<Feature, String>(){

			@Override
			public String apply(Feature feature){

				if(feature instanceof BinaryFeature){
					BinaryFeature binaryFeature = (BinaryFeature)feature;

					return (binaryFeature.getName()).getValue() + "=" + binaryFeature.getValue();
				}

				return (feature.getName()).getValue();
			}
		};

		List<String> ids = Lists.transform(features, function);

		if(ids.size() <= 10){
			return ids.toString();
		}

		List<String> result = new ArrayList<>();
		result.addAll(ids.subList(0, 2));
		result.add("...");
		result.addAll(ids.subList(ids.size() - 2, ids.size()));

		return result.toString();
	}
}