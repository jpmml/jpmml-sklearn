/*
 * Copyright (c) 2025 Villu Ruusmann
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

import org.dmg.pmml.DerivedField;
import org.dmg.pmml.Field;
import org.dmg.pmml.Lag;
import org.jpmml.converter.Feature;
import org.jpmml.converter.FeatureUtil;
import org.jpmml.converter.FieldNameUtil;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn.Transformer;

public class LagTransformer extends Transformer {

	public LagTransformer(String module, String name){
		super(module, name);
	}

	@Override
	public List<Feature> encodeFeatures(List<Feature> features, SkLearnEncoder encoder){
		Integer n = getN();

		List<Feature> result = new ArrayList<>();

		for(int i = 0; i < features.size(); i++){
			Feature feature = features.get(i);

			Field<?> field = feature.getField();

			Lag lag = new Lag(field.requireName())
				.setN(n);

			DerivedField derivedField = encoder.createDerivedField(FieldNameUtil.create("lag", feature, n), field.requireOpType(), field.requireDataType(), lag);

			feature = FeatureUtil.createFeature(derivedField, encoder);

			result.add(feature);
		}

		return result;
	}

	public Integer getN(){
		return getInteger("n");
	}
}