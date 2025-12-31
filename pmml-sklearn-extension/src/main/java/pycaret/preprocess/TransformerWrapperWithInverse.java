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

import com.google.common.collect.Iterables;
import org.dmg.pmml.DataField;
import org.dmg.pmml.Value;
import org.jpmml.converter.Feature;
import org.jpmml.converter.FeatureUtil;
import org.jpmml.converter.FieldUtil;
import org.jpmml.converter.Label;
import org.jpmml.converter.ResolutionException;
import org.jpmml.converter.ScalarLabel;
import org.jpmml.converter.ScalarLabelUtil;
import org.jpmml.converter.SchemaException;
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
		List<String> featureNames = getFeatureNames();

		LabelEncoder transformer = getTransformer();

		Label label = encoder.getLabel();
		if(label == null){
			throw new SchemaException("Expected a label, got no label");
		}

		List<Feature> result;

		if(features.isEmpty()){
			result = InitializerUtil.selectFeatures(featureNames, Collections.emptyList(), encoder);
		} else

		{
			result = features;
		}

		ScalarLabel scalarLabel = (ScalarLabel)label;

		Feature labelFeature = FeatureUtil.findLabelFeature(result, scalarLabel);
		if(labelFeature == null){
			throw new ResolutionException("Column \'" + scalarLabel.getName() + "\' not found in " + FeatureUtil.formatNames(result, '\''));
		}

		int labelFeatureIndex = result.indexOf(labelFeature);

		if(labelFeature instanceof WildcardFeature){
			WildcardFeature wildcardFeature = (WildcardFeature)labelFeature;

			DataField dataField = wildcardFeature.getField();

			FieldUtil.clearValues(dataField, Value.Property.VALID);

			Feature transformedLabelFeature = Iterables.getOnlyElement(transformer.encode(Collections.singletonList(labelFeature), encoder));

			result = new ArrayList<>(result);
			result.set(labelFeatureIndex, transformedLabelFeature);

			scalarLabel = ScalarLabelUtil.createScalarLabel(dataField);

			encoder.setLabel(scalarLabel);
		} else

		{
			throw new IllegalArgumentException();
		}

		return result;
	}

	@Override
	public LabelEncoder getTransformer(){
		return getTransformer("transformer", LabelEncoder.class);
	}
}