/*
 * Copyright (c) 2023 Villu Ruusmann
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
package pycaret.preprocess;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import org.dmg.pmml.Apply;
import org.dmg.pmml.DataType;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.OpType;
import org.dmg.pmml.PMMLFunctions;
import org.jpmml.converter.Feature;
import org.jpmml.converter.FeatureUtil;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn.MultiTransformer;

public class RareCategoryGrouping extends MultiTransformer {

	public RareCategoryGrouping(String module, String name){
		super(module, name);
	}

	@Override
	public OpType getOpType(){
		return OpType.CATEGORICAL;
	}

	@Override
	public List<Feature> encodeFeatures(List<Feature> features, SkLearnEncoder encoder){
		Map<String, ?> toOther = getToOther();
		Object value = getValue();

		List<Feature> result = new ArrayList<>();

		for(int i = 0; i < features.size(); i++){
			Feature feature = features.get(i);

			List<?> categories = (List<?>)toOther.get(feature.getName());
			if(categories == null){
				throw new IllegalArgumentException();
			} // End if

			if(!categories.isEmpty()){
				DataType dataType = feature.getDataType();

				Apply valueApply = PMMLUtil.createApply((categories.size() == 1 ? PMMLFunctions.EQUAL : PMMLFunctions.ISIN), feature.ref());

				for(Object category : categories){
					valueApply.addExpressions(PMMLUtil.createConstant(category, dataType));
				}

				Apply apply = PMMLUtil.createApply(PMMLFunctions.IF,
					valueApply,
					PMMLUtil.createConstant(value, dataType),
					feature.ref()
				);

				DerivedField derivedField = encoder.createDerivedField(createFieldName("rareCategoryGrouping", feature), OpType.CATEGORICAL, dataType, apply);

				feature = FeatureUtil.createFeature(derivedField, encoder);
			}

			result.add(feature);
		}

		return result;
	}

	public Map<String, ?> getToOther(){

		// PyCaret 3.0.0-RC
		if(containsKey("_to_other")){
			return getDict("_to_other");
		}

		// PyCaret 3.0.0+
		return getDict("to_other_");
	}

	public Object getValue(){
		return getScalar("value");
	}
}