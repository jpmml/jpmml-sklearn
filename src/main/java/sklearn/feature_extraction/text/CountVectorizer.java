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
package sklearn.feature_extraction.text;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;

import com.google.common.collect.BiMap;
import com.google.common.collect.HashBiMap;
import numpy.core.Scalar;
import org.dmg.pmml.Apply;
import org.dmg.pmml.DataType;
import org.dmg.pmml.DefineFunction;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.FieldRef;
import org.dmg.pmml.OpType;
import org.dmg.pmml.ParameterField;
import org.dmg.pmml.TextIndex;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.Feature;
import org.jpmml.converter.PMMLEncoder;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.converter.ValueUtil;
import org.jpmml.sklearn.ClassDictUtil;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn.HasNumberOfFeatures;
import sklearn.Transformer;

public class CountVectorizer extends Transformer implements HasNumberOfFeatures {

	public CountVectorizer(String module, String name){
		super(module, name);
	}

	@Override
	public int getNumberOfFeatures(){
		return 1;
	}

	@Override
	public OpType getOpType(){
		return OpType.CATEGORICAL;
	}

	@Override
	public DataType getDataType(){
		return DataType.STRING;
	}

	@Override
	public List<Feature> encodeFeatures(List<String> ids, List<Feature> features, SkLearnEncoder encoder){
		String analyzer = getAnalyzer();
		Boolean binary = getBinary();
		Boolean lowercase = getLowercase();
		String stripAccents = getStripAccents();
		String tokenPattern = getTokenPattern();
		Map<String, Scalar> vocabulary = getVocabulary();

		ClassDictUtil.checkSize(1, ids, features);

		Feature feature = features.get(0);

		switch(analyzer){
			case "word":
				break;
			default:
				throw new IllegalArgumentException(analyzer);
		}

		if(stripAccents != null){
			throw new IllegalArgumentException(stripAccents);
		} // End if

		if(tokenPattern != null && !("(?u)\\b\\w\\w+\\b").equals(tokenPattern)){
			throw new IllegalArgumentException();
		}

		BiMap<String, Integer> termIndexMap = HashBiMap.create(vocabulary.size());

		Collection<Map.Entry<String, Scalar>> entries = vocabulary.entrySet();
		for(Map.Entry<String, Scalar> entry : entries){
			termIndexMap.put(entry.getKey(), ValueUtil.asInt((Number)(entry.getValue()).getOnlyElement()));
		}

		BiMap<Integer, String> indexTermMap = termIndexMap.inverse();

		ParameterField documentField = new ParameterField(FieldName.create("document"));

		ParameterField termField = new ParameterField(FieldName.create("term"));

		TextIndex textIndex = new TextIndex(documentField.getName())
			.setLocalTermWeights(binary ? TextIndex.LocalTermWeights.BINARY : null)
			.setExpression(new FieldRef(termField.getName()));

		DefineFunction defineFunction = new DefineFunction("termFrequency", OpType.CONTINUOUS, null)
			.setDataType(DataType.DOUBLE)
			.addParameterFields(documentField, termField)
			.setExpression(textIndex);

		encoder.addDefineFunction(defineFunction);

		if(lowercase){
			DerivedField derivedField = encoder.createDerivedField(FieldName.create("lowercase(" + (feature.getName()).getValue() + ")"), OpType.CATEGORICAL, DataType.STRING, PMMLUtil.createApply("lowercase", feature.ref()));

			feature = new Feature(encoder, derivedField.getName(), derivedField.getDataType()){

				@Override
				public ContinuousFeature toContinuousFeature(){
					throw new UnsupportedOperationException();
				}
			};
		}

		ids.clear();

		List<Feature> result = new ArrayList<>();

		for(int i = 0, max = indexTermMap.size(); i < max; i++){
			String term = indexTermMap.get(i);

			final
			Apply apply = PMMLUtil.createApply(defineFunction.getName(), feature.ref(), PMMLUtil.createConstant(term));

			Feature termFrequencyFeature = new Feature(encoder, FieldName.create("tf(" + term + ")"), DataType.DOUBLE){

				@Override
				public ContinuousFeature toContinuousFeature(){
					PMMLEncoder encoder = ensureEncoder();

					DerivedField derivedField = encoder.getDerivedField(getName());
					if(derivedField == null){
						derivedField = encoder.createDerivedField(getName(), OpType.CONTINUOUS, getDataType(), apply);
					}

					return new ContinuousFeature(encoder, derivedField);
				}
			};

			ids.add((termFrequencyFeature.getName()).getValue());

			result.add(termFrequencyFeature);
		}

		return result;
	}

	public String getAnalyzer(){
		return (String)get("analyzer");
	}

	public Boolean getBinary(){
		return (Boolean)get("binary");
	}

	public Boolean getLowercase(){
		return (Boolean)get("lowercase");
	}

	public String getStripAccents(){
		return (String)get("strip_accents");
	}

	public String getTokenPattern(){
		return (String)get("token_pattern");
	}

	public Map<String, Scalar> getVocabulary(){
		return (Map)get("vocabulary_");
	}
}