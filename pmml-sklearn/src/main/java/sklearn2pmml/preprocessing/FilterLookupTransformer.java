/*
 * Copyright (c) 2021 Villu Ruusmann
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
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.OpType;
import org.dmg.pmml.PMMLFunctions;
import org.jpmml.converter.ExpressionUtil;
import org.jpmml.converter.Feature;
import org.jpmml.converter.FeatureUtil;
import org.jpmml.converter.SchemaUtil;
import org.jpmml.sklearn.SkLearnEncoder;

public class FilterLookupTransformer extends LookupTransformer {

	public FilterLookupTransformer(String module, String name){
		super(module, name);
	}

	@Override
	public List<Feature> encodeFeatures(List<Feature> features, SkLearnEncoder encoder){
		Feature feature = SchemaUtil.getOnlyFeature(features);

		List<Feature> mappedFeatures = super.encodeFeatures(features, encoder);

		Feature mappedFeature = SchemaUtil.getOnlyFeature(mappedFeatures);

		Apply apply = ExpressionUtil.createApply(PMMLFunctions.IF,
			ExpressionUtil.createApply(PMMLFunctions.ISNOTMISSING, mappedFeature.ref()),
			mappedFeature.ref(),
			feature.ref()
		);

		DerivedField derivedField = encoder.createDerivedField(createFieldName("filterLookup", features), OpType.CATEGORICAL, feature.getDataType(), apply);

		return Collections.singletonList(FeatureUtil.createFeature(derivedField, encoder));
	}

	@Override
	public Object getDefaultValue(){
		Object defaultValue = super.getDefaultValue();

		if(defaultValue != null){
			throw new IllegalArgumentException();
		}

		return defaultValue;
	}
}