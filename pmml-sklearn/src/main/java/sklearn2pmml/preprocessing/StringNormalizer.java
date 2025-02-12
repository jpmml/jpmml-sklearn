/*
 * Copyright (c) 2017 Villu Ruusmann
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

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import org.dmg.pmml.DataType;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.Expression;
import org.dmg.pmml.Field;
import org.dmg.pmml.OpType;
import org.dmg.pmml.PMMLFunctions;
import org.jpmml.converter.ExpressionUtil;
import org.jpmml.converter.Feature;
import org.jpmml.converter.StringFeature;
import org.jpmml.python.ClassDictUtil;
import org.jpmml.sklearn.SkLearnEncoder;

public class StringNormalizer extends StringTransformer {

	public StringNormalizer(String module, String name){
		super(module, name);
	}

	@Override
	public List<Feature> encodeFeatures(List<Feature> features, SkLearnEncoder encoder){
		String function = getFunction();
		Boolean trimBlanks = getTrimBlanks();

		if(function == null && !trimBlanks){
			return features;
		}

		ClassDictUtil.checkSize(1, features);

		Feature feature = features.get(0);

		Expression expression = feature.ref();

		if(function != null){
			expression = ExpressionUtil.createApply(translateFunction(function), expression);
		} // End if

		if(trimBlanks){
			expression = ExpressionUtil.createApply(PMMLFunctions.TRIMBLANKS, expression);
		}

		Field<?> field = encoder.toCategorical(feature.getName(), Collections.emptyList());

		// XXX: Should have been set by the previous transformer
		field.setDataType(DataType.STRING);

		DerivedField derivedField = encoder.createDerivedField(createFieldName("normalize", feature), OpType.CATEGORICAL, DataType.STRING, expression);

		return Collections.singletonList(new StringFeature(encoder, derivedField));
	}

	public String getFunction(){
		return getOptionalEnum("function", this::getOptionalString, Arrays.asList(StringNormalizer.FUNCTION_LOWER, StringNormalizer.FUNCTION_LOWERCASE, StringNormalizer.FUNCTION_UPPER, StringNormalizer.FUNCTION_UPPERCASE));
	}

	public Boolean getTrimBlanks(){
		return getOptionalBoolean("trim_blanks", Boolean.FALSE);
	}

	static
	private String translateFunction(String function){

		switch(function){
			// Python style
			case StringNormalizer.FUNCTION_LOWER:
			// PMML style
			case StringNormalizer.FUNCTION_LOWERCASE:
				return PMMLFunctions.LOWERCASE;
			// Python style
			case StringNormalizer.FUNCTION_UPPER:
			// PMML style
			case StringNormalizer.FUNCTION_UPPERCASE:
				return PMMLFunctions.UPPERCASE;
			default:
				throw new IllegalArgumentException(function);
		}
	}

	private static final String FUNCTION_LOWER = "lower";
	private static final String FUNCTION_LOWERCASE = "lowercase";
	private static final String FUNCTION_UPPER = "upper";
	private static final String FUNCTION_UPPERCASE = "uppercase";
}