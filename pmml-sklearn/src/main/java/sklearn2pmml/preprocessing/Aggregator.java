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

import java.util.Collections;
import java.util.List;

import org.dmg.pmml.Apply;
import org.dmg.pmml.DataType;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.OpType;
import org.dmg.pmml.PMMLFunctions;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.Feature;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn.HasNumberOfFeatures;
import sklearn.Transformer;

public class Aggregator extends Transformer {

	public Aggregator(String module, String name){
		super(module, name);
	}

	@Override
	public int getNumberOfFeatures(){
		return HasNumberOfFeatures.UNKNOWN;
	}

	@Override
	public List<Feature> encodeFeatures(List<Feature> features, SkLearnEncoder encoder){
		String function = getFunction();

		if(features.size() <= 1){
			return features;
		}

		Apply apply = PMMLUtil.createApply(translateFunction(function));

		for(Feature feature : features){
			apply.addExpressions(feature.ref());
		}

		DerivedField derivedField = encoder.createDerivedField(createFieldName(function, features), OpType.CONTINUOUS, DataType.DOUBLE, apply);

		return Collections.singletonList(new ContinuousFeature(encoder, derivedField));
	}

	public String getFunction(){
		return getString("function");
	}

	static
	private String translateFunction(String function){

		switch(function){
			case "max":
				return PMMLFunctions.MAX;
			case "mean":
			case "avg":
				return PMMLFunctions.AVG;
			case "min":
				return PMMLFunctions.MIN;
			case "prod":
			case "product":
				return PMMLFunctions.PRODUCT;
			case "sum":
				return PMMLFunctions.SUM;
			default:
				throw new IllegalArgumentException(function);
		}
	}
}