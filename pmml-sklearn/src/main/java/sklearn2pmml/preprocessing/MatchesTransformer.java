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
import org.dmg.pmml.OpType;
import org.dmg.pmml.PMMLFunctions;
import org.jpmml.converter.BooleanFeature;
import org.jpmml.converter.ExpressionUtil;
import org.jpmml.converter.Feature;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.python.ClassDictUtil;
import org.jpmml.sklearn.SkLearnEncoder;

public class MatchesTransformer extends RegExTransformer {

	public MatchesTransformer(){
		this("sklearn2pmml.preprocessing", "MatchesTransformer");
	}

	public MatchesTransformer(String module, String name){
		super(module, name);
	}

	@Override
	public List<Feature> encodeFeatures(List<Feature> features, SkLearnEncoder encoder){
		String pattern = getPattern();
		String reFlavour = getReFlavour();

		if(reFlavour != null){
			pattern = translatePattern(pattern, reFlavour);
		}

		ClassDictUtil.checkSize(1, features);

		Feature feature = features.get(0);

		if(feature.getDataType() != DataType.STRING){
			throw new IllegalArgumentException();
		}

		Apply apply = ExpressionUtil.createApply(PMMLFunctions.MATCHES, feature.ref(), ExpressionUtil.createConstant(DataType.STRING, pattern));

		if(reFlavour != null){
			apply.addExtensions(PMMLUtil.createExtension("re_flavour", reFlavour));
		}

		DerivedField derivedField = encoder.createDerivedField(createFieldName("matches", feature, formatArg(pattern)), OpType.CATEGORICAL, DataType.BOOLEAN, apply);

		return Collections.singletonList(new BooleanFeature(encoder, derivedField));
	}

	@Override
	MatchesTransformer setPattern(String pattern){
		return (MatchesTransformer)super.setPattern(pattern);
	}

	@Override
	MatchesTransformer setReFlavour(String reFlavour){
		return (MatchesTransformer)super.setReFlavour(reFlavour);
	}
}