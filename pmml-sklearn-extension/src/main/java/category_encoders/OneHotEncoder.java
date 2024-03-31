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
package category_encoders;

import java.util.ArrayList;
import java.util.List;

import org.jpmml.converter.BinaryFeature;
import org.jpmml.converter.CategoricalFeature;
import org.jpmml.converter.Feature;
import org.jpmml.converter.MissingValueFeature;
import org.jpmml.converter.ValueUtil;
import org.jpmml.sklearn.SkLearnEncoder;

public class OneHotEncoder extends BaseEncoder {

	public OneHotEncoder(String module, String name){
		super(module, name);
	}

	@Override
	public List<Feature> encodeFeatures(List<Feature> features, SkLearnEncoder encoder){
		Boolean dropInvariant = getDropInvariant();
		String handleMissing = getHandleMissing();
		String handleUnknown = getHandleUnknown();
		OrdinalEncoder ordinalEncoder = getOrdinalEncoder();

		if(dropInvariant){
			throw new IllegalArgumentException();
		}

		switch(handleMissing){
			case "error":
			case "return_nan":
			case "value":
			case "indicator":
				break;
			default:
				throw new IllegalArgumentException(handleMissing);
		} // End switch

		switch(handleUnknown){
			case "error":
			case "value":
				break;
			default:
				throw new IllegalArgumentException(handleUnknown);
		}

		features = ordinalEncoder.encode(features, encoder);

		List<Feature> result = new ArrayList<>();

		for(int i = 0; i < features.size(); i++){
			Feature feature = features.get(i);

			if(feature instanceof CategoricalFeature){
				CategoricalFeature categoricalFeature = (CategoricalFeature)feature;

				List<?> categories = categoricalFeature.getValues();

				for(int j = 0; j < categories.size(); j++){
					Object category = categories.get(j);

					if(ValueUtil.isNaN(category)){

						switch(handleMissing){
							case "value":
							case "indicator":
								result.add(new MissingValueFeature(encoder, categoricalFeature));
								break;
							default:
								break;
						}
					} else

					{
						result.add(new BinaryFeature(encoder, categoricalFeature, category));
					}
				}
			} else

			{
				throw new IllegalArgumentException();
			}
		}

		return result;
	}

	public OrdinalEncoder getOrdinalEncoder(){
		return get("ordinal_encoder", OrdinalEncoder.class);
	}
}