/*
 * Copyright (c) 2023 Villu Ruusmann
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
package sklearn2pmml.cross_reference;

import java.util.ArrayList;
import java.util.List;

import org.dmg.pmml.Field;
import org.jpmml.converter.Feature;
import org.jpmml.converter.FeatureUtil;
import org.jpmml.sklearn.SkLearnEncoder;

public class Recaller extends MemoryManager {

	public Recaller(String module, String name){
		super(module, name);
	}

	@Override
	public List<Feature> encodeFeatures(List<Feature> features, SkLearnEncoder encoder){
		List<String> names = getNames();

		List<Feature> result = new ArrayList<>();

		for(int i = 0; i < names.size(); i++){
			String name = names.get(i);

			Feature feature = encoder.recall(name);
			if(feature == null){
				Field<?> field = encoder.getField(name);

				feature = FeatureUtil.createFeature(field, encoder);
			}

			result.add(feature);
		}

		return result;
	}
}