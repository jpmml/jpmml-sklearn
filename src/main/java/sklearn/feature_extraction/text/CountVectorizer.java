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

import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.Reader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import com.google.common.base.Joiner;
import com.google.common.collect.BiMap;
import com.google.common.collect.HashBiMap;
import com.google.common.io.CharStreams;
import numpy.DType;
import numpy.core.ScalarUtil;
import org.dmg.pmml.Apply;
import org.dmg.pmml.Constant;
import org.dmg.pmml.DataType;
import org.dmg.pmml.DefineFunction;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.FieldRef;
import org.dmg.pmml.OpType;
import org.dmg.pmml.PMMLFunctions;
import org.dmg.pmml.ParameterField;
import org.dmg.pmml.TextIndex;
import org.dmg.pmml.TextIndexNormalization;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.Feature;
import org.jpmml.converter.FieldNameUtil;
import org.jpmml.converter.ObjectFeature;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.converter.StringFeature;
import org.jpmml.converter.ValueUtil;
import org.jpmml.python.ClassDictUtil;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn.Transformer;
import sklearn2pmml.feature_extraction.text.Splitter;

public class CountVectorizer extends Transformer {

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
	public List<Feature> encodeFeatures(List<Feature> features, SkLearnEncoder encoder){
		Boolean lowercase = getLowercase();
		Map<String, ?> vocabulary = getVocabulary();

		ClassDictUtil.checkSize(1, features);

		Feature feature = features.get(0);

		BiMap<String, Integer> termIndexMap = HashBiMap.create(vocabulary.size());

		Collection<? extends Map.Entry<String, ?>> entries = vocabulary.entrySet();
		for(Map.Entry<String, ?> entry : entries){
			termIndexMap.put(entry.getKey(), ValueUtil.asInteger((Number)ScalarUtil.decode(entry.getValue())));
		}

		BiMap<Integer, String> indexTermMap = termIndexMap.inverse();

		DType dtype = getDType();

		DataType dataType = (dtype != null ? dtype.getDataType() : DataType.DOUBLE);

		if(lowercase){
			Apply apply = PMMLUtil.createApply(PMMLFunctions.LOWERCASE, feature.ref());

			DerivedField derivedField = encoder.ensureDerivedField(FieldNameUtil.create("lowercase", feature), OpType.CATEGORICAL, DataType.STRING, () -> apply);

			feature = new StringFeature(encoder, derivedField);
		}

		DefineFunction defineFunction = encodeDefineFunction(feature);

		encoder.addDefineFunction(defineFunction);

		List<Feature> result = new ArrayList<>();

		for(int i = 0, max = indexTermMap.size(); i < max; i++){
			String term = indexTermMap.get(i);

			Apply apply = encodeApply(defineFunction.getName(), feature, i, term);

			Feature termFeature = new ObjectFeature(encoder, FieldNameUtil.create(functionName(), feature, term), dataType){

				@Override
				public ContinuousFeature toContinuousFeature(){
					return toContinuousFeature(getName(), getDataType(), () -> apply);
				}
			};

			result.add(termFeature);
		}

		return result;
	}

	public DefineFunction encodeDefineFunction(Feature feature){
		String analyzer = getAnalyzer();
		List<String> stopWords = getStopWords();
		Object[] nGramRange = getNGramRange();
		Boolean binary = getBinary();
		Object preprocessor = getPreprocessor();
		String stripAccents = getStripAccents();
		Splitter tokenizer = getTokenizer();

		switch(analyzer){
			case "word":
				break;
			default:
				throw new IllegalArgumentException(analyzer);
		}

		if(preprocessor != null){
			throw new IllegalArgumentException();
		} // End if

		if(stripAccents != null){
			throw new IllegalArgumentException(stripAccents);
		}

		ParameterField documentField = new ParameterField(FieldName.create("document"));

		ParameterField termField = new ParameterField(FieldName.create("term"));

		TextIndex textIndex = new TextIndex(documentField.getName(), new FieldRef(termField.getName()))
			.setTokenize(Boolean.TRUE)
			.setWordSeparatorCharacterRE(tokenizer.getSeparatorRE())
			.setLocalTermWeights(binary ? TextIndex.LocalTermWeights.BINARY : null);

		if((stopWords != null && stopWords.size() > 0) && !Arrays.equals(nGramRange, new Integer[]{1, 1})){
			Map<String, List<String>> data = new LinkedHashMap<>();
			data.put("string", Collections.singletonList("(^|\\s+)\\p{Punct}*(" + JOINER.join(stopWords) + ")\\p{Punct}*(\\s+|$)"));
			data.put("stem", Collections.singletonList(" "));
			data.put("regex", Collections.singletonList("true"));

			TextIndexNormalization textIndexNormalization = new TextIndexNormalization(null, PMMLUtil.createInlineTable(data))
				.setRecursive(Boolean.TRUE); // Handles consecutive matches. See http://stackoverflow.com/a/25085385

			textIndex.addTextIndexNormalizations(textIndexNormalization);
		}

		FieldName name = createFieldName(functionName(), feature);

		DefineFunction defineFunction = new DefineFunction(name.getValue(), OpType.CONTINUOUS, DataType.DOUBLE, null, textIndex)
			.addParameterFields(documentField, termField);

		return defineFunction;
	}

	public Apply encodeApply(String function, Feature feature, int index, String term){
		Constant constant = PMMLUtil.createConstant(term, DataType.STRING);

		return PMMLUtil.createApply(function, feature.ref(), constant);
	}

	public String functionName(){
		return "tf";
	}

	public String getAnalyzer(){
		return getString("analyzer");
	}

	public Boolean getBinary(){
		return getBoolean("binary");
	}

	public DType getDType(){
		return (DType)getDType(false);
	}

	public Boolean getLowercase(){
		return getBoolean("lowercase");
	}

	public Object[] getNGramRange(){
		return getTuple("ngram_range");
	}

	public Object getPreprocessor(){
		return getOptionalObject("preprocessor");
	}

	public List<String> getStopWords(){
		Object stopWords = getOptionalObject("stop_words");

		if(stopWords instanceof String){
			return loadStopWords((String)stopWords);
		}

		return (List)stopWords;
	}

	public String getStripAccents(){
		return getOptionalString("strip_accents");
	}

	public Splitter getTokenizer(){
		return get("tokenizer", Splitter.class);
	}

	public Map<String, ?> getVocabulary(){
		return getDict("vocabulary_");
	}

	static
	private List<String> loadStopWords(String stopWords){
		InputStream is = CountVectorizer.class.getResourceAsStream("/stop_words/" + stopWords + ".txt");

		if(is == null){
			throw new IllegalArgumentException(stopWords);
		}

		try(Reader reader = new InputStreamReader(is, "UTF-8")){
			return CharStreams.readLines(reader);
		} catch(IOException ioe){
			throw new IllegalArgumentException(stopWords, ioe);
		}
	}

	private static final Joiner JOINER = Joiner.on("|");
}
