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
package sklearn.preprocessing;

import java.util.ArrayList;
import java.util.List;

import org.dmg.pmml.Apply;
import org.dmg.pmml.DerivedField;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.Feature;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.converter.ValueUtil;
import org.jpmml.sklearn.ClassDictUtil;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn.Transformer;

public class MaxAbsScaler extends Transformer {

	public MaxAbsScaler(String module, String name){
		super(module, name);
	}

	@Override
	public List<Feature> encodeFeatures(List<String> ids, List<Feature> features, SkLearnEncoder encoder){
		List<? extends Number> scale = getScale();

		ClassDictUtil.checkSize(ids, features, scale);

		List<Feature> result = new ArrayList<>();

		for(int i = 0; i < features.size(); i++){
			String id = ids.get(i);
			Feature feature = features.get(i);

			Number value = scale.get(i);
			if(ValueUtil.isOne(value)){
				result.add(feature);

				continue;
			}

			// "$name / scale"
			Apply apply = PMMLUtil.createApply("/", (feature.toContinuousFeature()).ref(), PMMLUtil.createConstant(value));

			DerivedField derivedField = encoder.createDerivedField(createName(id), apply);

			result.add(new ContinuousFeature(encoder, derivedField));
		}

		return result;
	}

	public List<? extends Number> getScale(){
		return (List)ClassDictUtil.getArray(this, "scale_");
	}
}