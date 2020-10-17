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
package sklearn;

import java.util.List;

import org.jpmml.converter.Feature;
import org.jpmml.sklearn.SkLearnEncoder;

abstract
public class Initializer extends MultiTransformer {

	public Initializer(String module, String name){
		super(module, name);
	}

	abstract
	public List<Feature> initializeFeatures(SkLearnEncoder encoder);

	@Override
	public int getNumberOfFeatures(){
		return HasNumberOfFeatures.UNKNOWN;
	}

	@Override
	public List<Feature> encodeFeatures(List<Feature> features, SkLearnEncoder encoder){

		if(features.size() != 0){
			throw new IllegalArgumentException("Transformer \'" + getClassName() + "\' must be the first step of the pipeline");
		}

		return initializeFeatures(encoder);
	}
}