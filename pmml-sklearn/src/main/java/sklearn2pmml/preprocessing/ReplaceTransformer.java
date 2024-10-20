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
import org.jpmml.converter.ExpressionUtil;
import org.jpmml.converter.Feature;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.converter.StringFeature;
import org.jpmml.python.ClassDictUtil;
import org.jpmml.sklearn.SkLearnEncoder;

public class ReplaceTransformer extends RegExTransformer {

	public ReplaceTransformer(){
		this("sklearn2pmml.preprocessing", "ReplaceTransformer");
	}

	public ReplaceTransformer(String module, String name){
		super(module, name);
	}

	@Override
	public List<Feature> encodeFeatures(List<Feature> features, SkLearnEncoder encoder){
		String pattern = getPattern();
		String replacement = getReplacement();
		String reFlavour = getReFlavour();

		if(reFlavour != null){
			pattern = translatePattern(pattern, reFlavour);
			replacement = translateReplacement(replacement, reFlavour);
		}

		ClassDictUtil.checkSize(1, features);

		Feature feature = features.get(0);

		if(feature.getDataType() != DataType.STRING){
			throw new IllegalArgumentException();
		}

		Apply apply = ExpressionUtil.createApply(PMMLFunctions.REPLACE, feature.ref(), ExpressionUtil.createConstant(DataType.STRING, pattern), ExpressionUtil.createConstant(DataType.STRING, replacement));

		if(reFlavour != null){
			apply.addExtensions(PMMLUtil.createExtension("re_flavour", reFlavour));
		}

		DerivedField derivedField = encoder.createDerivedField(createFieldName("replace", feature, formatArg(pattern), formatArg(replacement)), OpType.CATEGORICAL, DataType.STRING, apply);

		return Collections.singletonList(new StringFeature(encoder, derivedField));
	}

	@Override
	ReplaceTransformer setPattern(String pattern){
		return (ReplaceTransformer)super.setPattern(pattern);
	}

	@Override
	ReplaceTransformer setReFlavour(String reFlavour){
		return (ReplaceTransformer)super.setReFlavour(reFlavour);
	}

	public String getReplacement(){
		return getString("replacement");
	}

	ReplaceTransformer setReplacement(String replacement){
		setattr("replacement", replacement);

		return this;
	}

	static
	public String translateReplacement(String replacement, String reFlavour){

		switch(reFlavour){
			case RegExTransformer.RE_FLAVOUR_PCRE:
			case RegExTransformer.RE_FLAVOUR_PCRE2:
				return replacement;
			case RegExTransformer.RE_FLAVOUR_RE:
				return replacement
					.replaceAll("\\$", "\\$\\$")
					.replaceAll("\\\\(\\d)", "\\$$1");
			default:
				throw new IllegalArgumentException(reFlavour);
		}
	}
}