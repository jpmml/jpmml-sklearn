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
package sklearn;

import java.util.List;

import org.jpmml.converter.Feature;
import org.jpmml.sklearn.FeatureMapper;

abstract
public class Selector extends Transformer implements HasNumberOfFeatures {

	public Selector(String module, String name){
		super(module, name);
	}

	abstract
	public List<Feature> selectFeatures(List<Feature> features);

	@Override
	public List<Feature> encodeFeatures(List<String> ids, List<Feature> features, FeatureMapper featureMapper){
		return selectFeatures(features);
	}
}