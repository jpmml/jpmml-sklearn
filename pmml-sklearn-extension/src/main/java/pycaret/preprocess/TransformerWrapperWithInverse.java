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
import java.util.List;
import java.util.stream.Collectors;

import com.google.common.collect.Iterables;
import org.dmg.pmml.DataField;
import org.dmg.pmml.Value;
import org.jpmml.converter.Feature;
import org.jpmml.converter.Label;
import org.jpmml.converter.ScalarLabel;
import org.jpmml.converter.WildcardFeature;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn.InitializerUtil;
import sklearn.ScalarLabelUtil;
import sklearn.preprocessing.LabelEncoder;
import sklearn2pmml.decoration.DomainUtil;

public class TransformerWrapperWithInverse extends TransformerWrapper {

	public TransformerWrapperWithInverse(String module, String name){
		super(module, name);
	}

	@Override
	public List<Feature> encodeFeatures(List<Feature> features, SkLearnEncoder encoder){
		List<String> featureNamesIn = getFeatureNamesIn();

		LabelEncoder transformer = getTransformer();

		Label label = encoder.getLabel();
		if(label == null){
			throw new IllegalArgumentException();
		}

		List<Feature> result;

		if(features.isEmpty()){
			result = InitializerUtil.selectFeatures(featureNamesIn, Collections.emptyList(), encoder);
		} else

		{
			result = features;
		}

		ScalarLabel scalarLabel = (ScalarLabel)label;

		Feature labelFeature = ScalarLabelUtil.findLabelFeature(scalarLabel, result);
		if(labelFeature == null){
			List<String> names = result.stream()
				.map(feature -> feature.getName())
				.collect(Collectors.toList());

			throw new IllegalArgumentException("Column \'" + scalarLabel.getName() + "\' not found in " + (names));
		}

		int labelFeatureIndex = result.indexOf(labelFeature);

		if(labelFeature instanceof WildcardFeature){
			WildcardFeature wildcardFeature = (WildcardFeature)labelFeature;

			DataField dataField = wildcardFeature.getField();

			DomainUtil.clearValues(dataField, Value.Property.VALID);

			Feature transformedLabelFeature = Iterables.getOnlyElement(transformer.encode(Collections.singletonList(labelFeature), encoder));

			result = new ArrayList<>(result);
			result.set(labelFeatureIndex, transformedLabelFeature);
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