/*
 * Copyright (c) 2022 Villu Ruusmann
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

import java.util.Collections;
import java.util.List;

import org.dmg.pmml.Apply;
import org.dmg.pmml.DataType;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.OpType;
import org.jpmml.converter.Feature;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.converter.SchemaUtil;
import org.jpmml.converter.StringFeature;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn.Transformer;

abstract
public class Formatter extends Transformer {

	public Formatter(String module, String name){
		super(module, name);
	}

	abstract
	public String getFunction();

	@Override
	public List<Feature> encodeFeatures(List<Feature> features, SkLearnEncoder encoder){
		String function = getFunction();
		String pattern = getPattern();

		SchemaUtil.checkSize(1, features);

		Feature feature = features.get(0);

		Apply apply = PMMLUtil.createApply(function, feature.ref(), PMMLUtil.createConstant(pattern, DataType.STRING));

		DerivedField derivedField = encoder.createDerivedField(createFieldName("format", feature, pattern), OpType.CATEGORICAL, DataType.STRING, apply);

		return Collections.singletonList(new StringFeature(encoder, derivedField));
	}

	public String getPattern(){
		return getString("pattern");
	}
}