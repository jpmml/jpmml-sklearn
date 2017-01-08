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

import org.dmg.pmml.DerivedField;
import org.dmg.pmml.Expression;
import org.dmg.pmml.FieldRef;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.Feature;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.converter.ValueUtil;
import org.jpmml.sklearn.ClassDictUtil;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn.Transformer;

public class MinMaxScaler extends Transformer {

	public MinMaxScaler(String module, String name){
		super(module, name);
	}

	@Override
	public List<Feature> encodeFeatures(List<String> ids, List<Feature> features, SkLearnEncoder encoder){
		List<? extends Number> min = getMin();
		List<? extends Number> scale = getScale();

		ClassDictUtil.checkSize(ids, features, min, scale);

		List<Feature> result = new ArrayList<>();

		for(int i = 0; i < features.size(); i++){
			String id = ids.get(i);
			Feature feature = features.get(i);

			Number minValue = min.get(i);
			Number scaleValue = scale.get(i);

			// "($name * scale) + min"
			Expression expression = (feature.toContinuousFeature()).ref();

			if(!ValueUtil.isOne(scaleValue)){
				expression = PMMLUtil.createApply("*", expression, PMMLUtil.createConstant(scaleValue));
			} // End if

			if(!ValueUtil.isZero(minValue)){
				expression = PMMLUtil.createApply("+", expression, PMMLUtil.createConstant(minValue));
			} // End if

			if(expression instanceof FieldRef){
				result.add(feature);

				continue;
			}

			DerivedField derivedField = encoder.createDerivedField(createName(id), expression);

			result.add(new ContinuousFeature(encoder, derivedField));
		}

		return result;
	}

	public List<? extends Number> getMin(){
		return (List)ClassDictUtil.getArray(this, "min_");
	}

	public List<? extends Number> getScale(){
		return (List)ClassDictUtil.getArray(this, "scale_");
	}
}