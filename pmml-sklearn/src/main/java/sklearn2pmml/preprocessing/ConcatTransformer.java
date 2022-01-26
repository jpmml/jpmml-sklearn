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

import java.util.Collections;
import java.util.List;

import org.dmg.pmml.Apply;
import org.dmg.pmml.DataType;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.Expression;
import org.dmg.pmml.OpType;
import org.dmg.pmml.PMMLFunctions;
import org.jpmml.converter.Feature;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.converter.StringFeature;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn.Transformer;

public class ConcatTransformer extends Transformer {

	public ConcatTransformer(String module, String name){
		super(module, name);
	}

	@Override
	public List<Feature> encodeFeatures(List<Feature> features, SkLearnEncoder encoder){
		String separator = getSeparator();

		Apply apply = PMMLUtil.createApply(PMMLFunctions.CONCAT);

		List<Expression> expressions = apply.getExpressions();

		for(int i = 0; i < features.size(); i++){
			Feature feature = features.get(i);

			if((i > 0) && !("").equals(separator)){
				expressions.add(PMMLUtil.createConstant(separator, DataType.STRING));
			}

			expressions.add(feature.ref());
		}

		DerivedField derivedField = encoder.createDerivedField(createFieldName("concat", features), OpType.CATEGORICAL, DataType.STRING, apply);

		return Collections.singletonList(new StringFeature(encoder, derivedField));
	}

	public String getSeparator(){
		return getString("separator");
	}
}