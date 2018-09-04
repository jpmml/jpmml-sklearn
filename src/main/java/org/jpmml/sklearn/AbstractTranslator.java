/*
 * Copyright (c) 2018 Villu Ruusmann
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

import java.util.List;

import org.dmg.pmml.FieldName;
import org.jpmml.converter.Feature;

abstract
public class AbstractTranslator {

	private List<? extends Feature> features = null;


	public Feature getFeature(int index){
		List<? extends Feature> features = getFeatures();

		if(index >= 0 && index < features.size()){
			return features.get(index);
		}

		throw new IllegalArgumentException(String.valueOf(index));
	}

	public Feature getFeature(FieldName name){
		List<? extends Feature> features = getFeatures();

		for(Feature feature : features){

			if((feature.getName()).equals(name)){
				return feature;
			}
		}

		throw new IllegalArgumentException(name.getValue());
	}

	public List<? extends Feature> getFeatures(){
		return this.features;
	}

	void setFeatures(List<? extends Feature> features){
		this.features = features;
	}
}