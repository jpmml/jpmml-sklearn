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
package sklearn.feature_extraction;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import org.dmg.pmml.DataField;
import org.dmg.pmml.DataType;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.OpType;
import org.jpmml.converter.BinaryFeature;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.Feature;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn.Initializer;

public class DictVectorizer extends Initializer {

	public DictVectorizer(String module, String name){
		super(module, name);
	}

	@Override
	public List<Feature> initializeFeatures(SkLearnEncoder encoder){
		List<String> featureNames = getFeatureNames();
		String separator = getSeparator();
		Map<String, Integer> vocabulary = getVocabulary();

		Feature[] featureArray = new Feature[featureNames.size()];

		for(String featureName : featureNames){
			String key = featureName;
			String value = null;

			int index = featureName.indexOf(separator);
			if(index > -1){
				key = featureName.substring(0, index);
				value = featureName.substring(index + separator.length());
			}

			FieldName name = FieldName.create(key);

			DataField dataField = encoder.getDataField(name);
			if(dataField == null){

				if(value != null){
					dataField = encoder.createDataField(name, OpType.CATEGORICAL, DataType.STRING);
				} else

				{
					dataField = encoder.createDataField(name, OpType.CONTINUOUS, DataType.DOUBLE);
				}
			}

			Feature feature;

			if(value != null){
				PMMLUtil.addValues(dataField, Collections.singletonList(value));

				feature = new BinaryFeature(encoder, dataField, value);
			} else

			{
				feature = new ContinuousFeature(encoder, dataField);
			}

			featureArray[vocabulary.get(featureName)] = feature;
		}

		List<Feature> result = new ArrayList<>();
		result.addAll(Arrays.asList(featureArray));

		return result;
	}

	public List<String> getFeatureNames(){
		return (List)get("feature_names_");
	}

	public String getSeparator(){
		return (String)get("separator");
	}

	public Map<String, Integer> getVocabulary(){
		return (Map)get("vocabulary_");
	}
}