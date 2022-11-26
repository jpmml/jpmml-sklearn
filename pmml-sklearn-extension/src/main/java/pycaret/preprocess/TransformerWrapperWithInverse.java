/*
 * Copyright (c) 2022 Villu Ruusmann
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
import java.util.Collections;
import java.util.Iterator;
import java.util.List;

import com.google.common.collect.Iterables;
import org.dmg.pmml.DataField;
import org.dmg.pmml.Value;
import org.jpmml.converter.Feature;
import org.jpmml.converter.WildcardFeature;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn.InitializerUtil;
import sklearn.preprocessing.LabelEncoder;

public class TransformerWrapperWithInverse extends TransformerWrapper {

	public TransformerWrapperWithInverse(String module, String name){
		super(module, name);
	}

	@Override
	public List<Feature> encodeFeatures(List<Feature> features, SkLearnEncoder encoder){
		List<String> featureNamesIn = getFeatureNamesIn();

		LabelEncoder transformer = getTransformer();

		if(!features.isEmpty()){
			throw new IllegalArgumentException();
		}

		List<Feature> result = InitializerUtil.selectFeatures(featureNamesIn, Collections.emptyList(), encoder);

		int labelIndex = (result.size() - 1);

		Feature labelFeature = result.get(labelIndex);

		if(labelFeature instanceof WildcardFeature){
			WildcardFeature wildcardFeature = (WildcardFeature)labelFeature;

			DataField dataField = wildcardFeature.getField();

			if(dataField.hasValues()){
				List<Value> pmmlValues = dataField.getValues();

				for(Iterator<Value> it = pmmlValues.iterator(); it.hasNext(); ){
					Value pmmlValue = it.next();

					Value.Property property = pmmlValue.getProperty();
					switch(property){
						case VALID:
							it.remove();
						case INVALID:
						case MISSING:
							break;
						default:
							break;
					}
				}
			}

			Feature transformedLabelFeature = Iterables.getOnlyElement(transformer.encode(Collections.singletonList(labelFeature), encoder));

			result = new ArrayList<>(result);
			result.set(labelIndex, transformedLabelFeature);
		} else

		{
			throw new IllegalArgumentException();
		}

		return result;
	}

	@Override
	public LabelEncoder getTransformer(){
		return get("transformer", LabelEncoder.class);
	}
}