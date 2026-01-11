/*
 * Copyright (c) 2020 Villu Ruusmann
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
import java.util.List;

import org.dmg.pmml.DataType;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.OpType;
import org.dmg.pmml.PMMLFunctions;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.ExpressionUtil;
import org.jpmml.converter.Feature;
import org.jpmml.converter.ObjectFeature;
import org.jpmml.converter.UnsupportedFeatureException;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn.Transformer;

public class SecondsSinceMidnightTransformer extends Transformer {

	public SecondsSinceMidnightTransformer(String module, String name){
		super(module, name);
	}

	@Override
	public DataType getDataType(){
		return DataType.INTEGER;
	}

	@Override
	public List<Feature> encodeFeatures(List<Feature> features, SkLearnEncoder encoder){
		List<Feature> result = new ArrayList<>();

		for(int i = 0; i < features.size(); i++){
			Feature feature = features.get(i);

			if(!(feature instanceof ObjectFeature)){
				throw new UnsupportedFeatureException("Expected a time-type object feature, got " + feature.typeString());
			}

			ObjectFeature objectFeature = (ObjectFeature)feature;

			DerivedField derivedField = encoder.ensureDerivedField(createFieldName("secondsSinceMidnight", objectFeature), OpType.CONTINUOUS, DataType.INTEGER, () -> ExpressionUtil.createApply(PMMLFunctions.DATESECONDSSINCEMIDNIGHT, objectFeature.ref()));

			result.add(new ContinuousFeature(encoder, derivedField));
		}

		return result;
	}
}