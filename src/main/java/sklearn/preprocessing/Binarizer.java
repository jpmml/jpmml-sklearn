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

import java.util.Collections;
import java.util.List;

import org.dmg.pmml.Apply;
import org.dmg.pmml.DerivedField;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.Feature;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn.Transformer;

public class Binarizer extends Transformer {

	public Binarizer(String module, String name){
		super(module, name);
	}

	@Override
	public List<Feature> encodeFeatures(List<String> ids, List<Feature> features, SkLearnEncoder encoder){
		Number threshold = getThreshold();

		if(ids.size() != 1 || features.size() != 1){
			throw new IllegalArgumentException();
		}

		String id = ids.get(0);
		Feature feature = features.get(0);

		// "($name <= threshold) ? 0 : 1"
		Apply apply = PMMLUtil.createApply("threshold", (feature.toContinuousFeature()).ref(), PMMLUtil.createConstant(threshold));

		DerivedField derivedField = encoder.createDerivedField(createName(id), apply);

		return Collections.<Feature>singletonList(new ContinuousFeature(encoder, derivedField));
	}

	public Number getThreshold(){
		return (Number)get("threshold");
	}
}