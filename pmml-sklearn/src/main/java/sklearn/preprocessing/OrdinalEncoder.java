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
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

import org.dmg.pmml.DataType;
import org.dmg.pmml.InvalidValueTreatmentMethod;
import org.jpmml.converter.Feature;
import org.jpmml.converter.InvalidValueDecorator;
import org.jpmml.converter.ValueUtil;
import org.jpmml.python.ClassDictUtil;
import org.jpmml.python.TypeInfo;
import org.jpmml.sklearn.SkLearnEncoder;

public class OrdinalEncoder extends BaseEncoder {

	public OrdinalEncoder(String module, String name){
		super(module, name);
	}

	@Override
	public List<Feature> encodeFeatures(List<Feature> features, SkLearnEncoder encoder){
		List<List<Object>> categories = getCategories();
		TypeInfo dtype = getDType();
		String handleUnknown = getHandleUnknown();
		Number unknownValue = null;

		ClassDictUtil.checkSize(categories, features);

		if(handleUnknown != null){

			if(Objects.equals(OrdinalEncoder.HANDLEUNKNOWN_USE_ENCODED_VALUE, handleUnknown)){
				unknownValue = getUnknownValue();

				if(ValueUtil.isNaN(unknownValue)){
					unknownValue = null;
				}
			}
		}

		List<Feature> result = new ArrayList<>();

		DataType dataType = dtype.getDataType();

		for(int i = 0; i < features.size(); i++){
			Feature feature = features.get(i);

			List<Object> featureCategories = new ArrayList<>(categories.get(i));

			encoder.toCategorical(feature.getName(), EncoderUtil.filterCategories(featureCategories));

			if(handleUnknown != null){
				InvalidValueTreatmentMethod invalidValueTreatmentMethod;

				switch(handleUnknown){
					case OrdinalEncoder.HANDLEUNKNOWN_ERROR:
						invalidValueTreatmentMethod = InvalidValueTreatmentMethod.RETURN_INVALID;
						break;
					case OrdinalEncoder.HANDLEUNKNOWN_USE_ENCODED_VALUE:
						invalidValueTreatmentMethod = InvalidValueTreatmentMethod.AS_IS;
						break;
					default:
						throw new IllegalArgumentException(handleUnknown);
				}

				EncoderUtil.addDecorator(feature, new InvalidValueDecorator(invalidValueTreatmentMethod, null));
			}

			result.add(EncoderUtil.encodeIndexFeature(this, feature, featureCategories, null, unknownValue, dataType, encoder));
		}

		return result;
	}

	public TypeInfo getDType(){
		return getDType("dtype", false);
	}

	public String getHandleUnknown(){
		return getEnum("handle_unknown", this::getOptionalString, Arrays.asList(OrdinalEncoder.HANDLEUNKNOWN_ERROR, OrdinalEncoder.HANDLEUNKNOWN_USE_ENCODED_VALUE));
	}

	public Number getUnknownValue(){
		return getNumber("unknown_value");
	}

	private static final String HANDLEUNKNOWN_ERROR = "error";
	private static final String HANDLEUNKNOWN_USE_ENCODED_VALUE = "use_encoded_value";
}