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
import java.util.Objects;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import org.dmg.pmml.Field;
import org.jpmml.converter.Feature;
import org.jpmml.python.ClassDictUtil;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn.Initializer;
import sklearn.InitializerUtil;
import sklearn.Selector;
import sklearn.Transformer;
import sklearn.preprocessing.Scaler;

public class TransformerWrapper extends Initializer {

	public TransformerWrapper(String module, String name){
		super(module, name);
	}

	@Override
	public List<Feature> initializeFeatures(SkLearnEncoder encoder){
		return encodeFeatures(Collections.emptyList(), encoder);
	}

	@Override
	public List<Feature> encodeFeatures(List<Feature> features, SkLearnEncoder encoder){
		List<String> featureNamesIn = getFeatureNamesIn();
		List<String> include = getInclude();

		Transformer transformer = getTransformer();

		if(features.isEmpty()){
			features = InitializerUtil.selectFeatures(featureNamesIn, features, encoder);
		}

		List<Feature> includeFeatures = new ArrayList<>();

		for(int i = 0; i < include.size(); i++){
			String includeColumn = include.get(i);

			Feature includeFeature;

			if(!features.isEmpty()){
				int index = featureNamesIn.indexOf(includeColumn);

				includeFeature = features.get(index);
			} else

			{
				includeFeature = InitializerUtil.selectFeature(includeColumn, features, encoder);
			}

			includeFeatures.add(includeFeature);
		}

		List<Feature> transformedIncludeFeatures = transformer.encode(includeFeatures, encoder);

		if(transformer instanceof Selector){
			Selector selector = (Selector)transformer;

			return transformedIncludeFeatures;
		} else

		if(transformer instanceof VariableSelector){
			VariableSelector variableSelector = (VariableSelector)transformer;

			return transformedIncludeFeatures;
		} else

		if(transformer instanceof Scaler){
			Scaler scaler = (Scaler)transformer;

			return transformedIncludeFeatures;
		} else

		{
			List<List<Feature>> transformedIncludeFeatureGroups = groupByField(transformedIncludeFeatures);

			ClassDictUtil.checkSize(includeFeatures, transformedIncludeFeatureGroups);

			List<Object> result = new ArrayList<>(features);

			for(int i = 0; i < include.size(); i++){
				String includeColumn = include.get(i);

				int index = featureNamesIn.indexOf(includeColumn);

				List<Feature> transformedIncludeFeatureGroup = transformedIncludeFeatureGroups.get(i);

				result.set(index, transformedIncludeFeatureGroup);
			}

			return result.stream()
				.flatMap(element -> {

					if(element instanceof List){
						List<Feature> featureGroup = (List<Feature>)element;

						return featureGroup.stream();
					} else

					{
						Feature feature = (Feature)element;

						return Stream.of(feature);
					}
				})
				.collect(Collectors.toList());
		}
	}

	public List<String> getFeatureNamesIn(){
		return getList("_feature_names_in", String.class);
	}

	public List<String> getExclude(){
		return getList("_exclude", String.class);
	}

	public List<String> getInclude(){
		return getList("_include", String.class);
	}

	public Transformer getTransformer(){
		return get("transformer", Transformer.class);
	}

	static
	private List<List<Feature>> groupByField(List<Feature> features){
		List<List<Feature>> result = new ArrayList<>();

		Field<?> prevField = null;

		List<Feature> fieldFeatures = null;

		for(Feature feature : features){
			Field<?> field = feature.getField();

			if(!Objects.equals(field, prevField)){
				fieldFeatures = new ArrayList<>();
				fieldFeatures.add(feature);

				result.add(fieldFeatures);
			} else

			{
				fieldFeatures.add(feature);
			}

			prevField = field;
		}

		return result;
	}
}