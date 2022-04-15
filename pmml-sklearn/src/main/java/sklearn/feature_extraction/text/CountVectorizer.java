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

import com.google.common.collect.BiMap;
import com.google.common.collect.HashBiMap;
import com.google.common.io.CharStreams;
import numpy.core.ScalarUtil;
import org.dmg.pmml.Apply;
import org.dmg.pmml.Constant;
import org.dmg.pmml.DataType;
import org.dmg.pmml.DefineFunction;
import org.dmg.pmml.DerivedField;
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
import org.jpmml.python.TypeInfo;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn.Transformer;
import sklearn2pmml.feature_extraction.text.Matcher;

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

		TypeInfo dtype = getDType();

		DataType dataType = (dtype != null ? dtype.getDataType() : DataType.DOUBLE);

		if(lowercase){
			Apply apply = PMMLUtil.createApply(PMMLFunctions.LOWERCASE, feature.ref());

			DerivedField derivedField = encoder.ensureDerivedField(FieldNameUtil.create(PMMLFunctions.LOWERCASE, feature), OpType.CATEGORICAL, DataType.STRING, () -> apply);

			feature = new StringFeature(encoder, derivedField);
		}

		DefineFunction defineFunction = encodeDefineFunction(feature, encoder);

		encoder.addDefineFunction(defineFunction);

		List<Feature> result = new ArrayList<>();

		for(int i = 0, max = indexTermMap.size(); i < max; i++){
			String term = indexTermMap.get(i);

			Apply apply = encodeApply(defineFunction, feature, i, term);

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

	public DefineFunction encodeDefineFunction(Feature feature, SkLearnEncoder encoder){
		String analyzer = getAnalyzer();
		List<String> stopWords = getStopWords();
		Object[] nGramRange = getNGramRange();
		Boolean binary = getBinary();
		Object preprocessor = getPreprocessor();
		String stripAccents = getStripAccents();
		Tokenizer tokenizer = getTokenizer();

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
		} // End if

		if(tokenizer == null){
			String tokenPattern = getTokenPattern();

			tokenizer = new Matcher()
				.setWordRE(tokenPattern);
		}

		ParameterField documentField = new ParameterField("document");

		ParameterField termField = new ParameterField("term");

		TextIndex textIndex = new TextIndex(documentField, new FieldRef(termField))
			.setLocalTermWeights(binary ? TextIndex.LocalTermWeights.BINARY : null);

		textIndex = tokenizer.configure(textIndex);

		stopWords:
		if((stopWords != null && !stopWords.isEmpty()) && !Arrays.equals(nGramRange, new Integer[]{1, 1})){
			String stopWordsRE = tokenizer.formatStopWordsRE(stopWords);

			if(stopWordsRE == null){
				break stopWords;
			}

			Map<String, List<String>> data = new LinkedHashMap<>();
			data.put("string", Collections.singletonList(stopWordsRE));
			data.put("stem", Collections.singletonList(" "));
			data.put("regex", Collections.singletonList("true"));

			TextIndexNormalization textIndexNormalization = new TextIndexNormalization(PMMLUtil.createInlineTable(data))
				.setRecursive(Boolean.TRUE); // Handles consecutive matches. See http://stackoverflow.com/a/25085385

			textIndex.addTextIndexNormalizations(textIndexNormalization);
		}

		String name = createFieldName(functionName(), feature);

		DefineFunction defineFunction = new DefineFunction(name, OpType.CONTINUOUS, DataType.INTEGER, null, textIndex)
			.addParameterFields(documentField, termField);

		return defineFunction;
	}

	public Apply encodeApply(DefineFunction defineFunction, Feature feature, int index, String term){
		Constant constant = PMMLUtil.createConstant(term, DataType.STRING);

		return PMMLUtil.createApply(defineFunction, feature.ref(), constant);
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

	public TypeInfo getDType(){
		return getDType("dtype", false);
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

	public Tokenizer getTokenizer(){
		return getOptional("tokenizer", Tokenizer.class);
	}

	/**
	 * @see CountVectorizer#TOKEN_PATTERN
	 */
	public String getTokenPattern(){
		return getString("token_pattern");
	}

	public Map<String, ?> getVocabulary(){
		return getDict("vocabulary_");
	}

	static
	private List<String> loadStopWords(String stopWords){
		InputStream is = (CountVectorizer.class).getResourceAsStream("/stop_words/" + stopWords + ".txt");

		if(is == null){
			throw new IllegalArgumentException(stopWords);
		}

		try(Reader reader = new InputStreamReader(is, "UTF-8")){
			return CharStreams.readLines(reader);
		} catch(IOException ioe){
			throw new IllegalArgumentException(stopWords, ioe);
		}
	}

	public static final String TOKEN_PATTERN = "(?u)\\b\\w\\w+\\b";
}
