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

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import org.dmg.pmml.DataType;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.Expression;
import org.dmg.pmml.Field;
import org.dmg.pmml.OpType;
import org.dmg.pmml.PMMLFunctions;
import org.jpmml.converter.Feature;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.converter.StringFeature;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn.Transformer;

public class StringNormalizer extends Transformer {

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

		List<Feature> result = new ArrayList<>();

		for(Feature feature : features){
			Expression expression = feature.ref();

			if(function != null){
				expression = PMMLUtil.createApply(translateFunction(function), expression);
			} // End if

			if(trimBlanks){
				expression = PMMLUtil.createApply(PMMLFunctions.TRIMBLANKS, expression);
			}

			Field<?> field = encoder.toCategorical(feature.getName(), Collections.emptyList());

			// XXX: Should have been set by the previous transformer
			field.setDataType(DataType.STRING);

			DerivedField derivedField = encoder.createDerivedField(createFieldName("normalize", feature), OpType.CATEGORICAL, DataType.STRING, expression);

			feature = new StringFeature(encoder, derivedField);

			result.add(feature);
		}

		return result;
	}

	public String getFunction(){
		return getOptionalString("function");
	}

	public Boolean getTrimBlanks(){
		return getOptionalBoolean("trim_blanks", Boolean.FALSE);
	}

	static
	private String translateFunction(String function){

		switch(function){
			case "lower":
			case "lowercase":
				return PMMLFunctions.LOWERCASE;
			case "upper":
			case "uppercase":
				return PMMLFunctions.UPPERCASE;
			default:
				throw new IllegalArgumentException(function);
		}

	}
}