/*
 * Copyright (c) 2019 Villu Ruusmann
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

import org.jpmml.converter.Feature;
import org.jpmml.python.TypeInfo;
import org.jpmml.sklearn.SkLearnEncoder;

public class CastTransformer extends TypeTransformer {

	public CastTransformer(String module, String name){
		super(module, name);
	}

	@Override
	public List<Feature> encodeFeatures(List<Feature> features, SkLearnEncoder encoder){
		TypeInfo dtype = getDType();

		List<Feature> result = new ArrayList<>();

		for(int i = 0; i < features.size(); i++){
			Feature feature = features.get(i);

			feature = refineFeature(feature, dtype, encoder);

			result.add(feature);
		}

		return result;
	}

	public TypeInfo getDType(){

		// SkLearn2PMML 0.101.0+
		if(hasattr("dtype_")){
			return getDType("dtype_");
		} else

		// SkLearn2PMML 0.100.2
		{
			return getDType("dtype");
		}
	}
}