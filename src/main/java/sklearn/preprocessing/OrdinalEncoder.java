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
package sklearn.preprocessing;

import java.util.ArrayList;
import java.util.List;

import numpy.DType;
import org.dmg.pmml.DataType;
import org.jpmml.converter.Feature;
import org.jpmml.converter.ValueUtil;
import org.jpmml.python.ClassDictUtil;
import org.jpmml.sklearn.SkLearnEncoder;

public class OrdinalEncoder extends BaseEncoder {

	public OrdinalEncoder(String module, String name){
		super(module, name);
	}

	@Override
	public List<Feature> encodeFeatures(List<Feature> features, SkLearnEncoder encoder){
		List<List<?>> categories = getCategories();
		DType dtype = getDType();
		String handleUnknown = getHandleUnknown();
		Number unknownValue = null;

		ClassDictUtil.checkSize(categories, features);

		if(handleUnknown != null){

			switch(handleUnknown){
				case "error":
					break;
				case "use_encoded_value":
					unknownValue = getUnknownValue();

					if(ValueUtil.isNaN(unknownValue)){
						unknownValue = null;
					}
					break;
				default:
					throw new IllegalArgumentException(handleUnknown);
			}
		}

		List<Feature> result = new ArrayList<>();

		DataType dataType = dtype.getDataType();

		for(int i = 0; i < features.size(); i++){
			Feature feature = features.get(i);
			List<?> featureCategories = categories.get(i);

			result.add(EncoderUtil.encodeIndexFeature(this, feature, featureCategories, null, unknownValue, dataType, encoder));
		}

		return result;
	}

	public Number getUnknownValue(){
		return getNumber("unknown_value");
	}
}