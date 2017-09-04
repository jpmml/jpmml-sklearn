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
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

import javax.xml.parsers.DocumentBuilder;

import com.google.common.base.Joiner;
import com.google.common.collect.BiMap;
import com.google.common.collect.HashBiMap;
import com.google.common.io.CharStreams;
import numpy.DType;
import numpy.core.Scalar;
import org.dmg.pmml.Apply;
import org.dmg.pmml.Constant;
import org.dmg.pmml.DataType;
import org.dmg.pmml.DefineFunction;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.FieldRef;
import org.dmg.pmml.InlineTable;
import org.dmg.pmml.OpType;
import org.dmg.pmml.ParameterField;
import org.dmg.pmml.TextIndex;
import org.dmg.pmml.TextIndexNormalization;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.DOMUtil;
import org.jpmml.converter.Feature;
import org.jpmml.converter.FeatureUtil;
import org.jpmml.converter.PMMLEncoder;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.converter.ValueUtil;
import org.jpmml.sklearn.ClassDictUtil;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn.HasNumberOfFeatures;
import sklearn.Transformer;
import sklearn2pmml.feature_extraction.text.Splitter;

public class CountVectorizer extends Transformer implements HasNumberOfFeatures {

	public CountVectorizer(String module, String name){
		super(module, name);
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
	public int getNumberOfFeatures(){
		return 1;
	}

	@Override
	public List<Feature> encodeFeatures(List<Feature> features, SkLearnEncoder encoder){
		Boolean lowercase = getLowercase();
		Map<String, Scalar> vocabulary = getVocabulary();

		ClassDictUtil.checkSize(1, features);

		Feature feature = features.get(0);

		BiMap<String, Integer> termIndexMap = HashBiMap.create(vocabulary.size());

		Collection<Map.Entry<String, Scalar>> entries = vocabulary.entrySet();
		for(Map.Entry<String, Scalar> entry : entries){
			termIndexMap.put(entry.getKey(), ValueUtil.asInt((Number)(entry.getValue()).getOnlyElement()));
		}

		BiMap<Integer, String> indexTermMap = termIndexMap.inverse();

		DType dtype = getDType();

		if(lowercase){
			FieldName name = FeatureUtil.createName("lowercase", feature);

			DerivedField derivedField = encoder.getDerivedField(name);
			if(derivedField == null){
				Apply apply = PMMLUtil.createApply("lowercase", feature.ref());

				derivedField = encoder.createDerivedField(name, OpType.CATEGORICAL, DataType.STRING, apply);
			}

			feature = new Feature(encoder, derivedField.getName(), derivedField.getDataType()){

				@Override
				public ContinuousFeature toContinuousFeature(){
					throw new UnsupportedOperationException();
				}
			};
		}

		DefineFunction defineFunction = encodeDefineFunction();

		encoder.addDefineFunction(defineFunction);

		List<Feature> result = new ArrayList<>();

		for(int i = 0, max = indexTermMap.size(); i < max; i++){
			String term = indexTermMap.get(i);

			final
			Apply apply = encodeApply(defineFunction.getName(), feature, i, term);

			Feature termFeature = new Feature(encoder, FieldName.create(defineFunction.getName() + "(" + term + ")"), dtype != null ? dtype.getDataType() : DataType.DOUBLE){

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

			result.add(termFeature);
		}

		return result;
	}

	public DefineFunction encodeDefineFunction(){
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

		TextIndex textIndex = new TextIndex(documentField.getName())
			.setTokenize(Boolean.TRUE)
			.setWordSeparatorCharacterRE(tokenizer.getSeparatorRE())
			.setLocalTermWeights(binary ? TextIndex.LocalTermWeights.BINARY : null)
			.setExpression(new FieldRef(termField.getName()));

		if((stopWords != null && stopWords.size() > 0) && !Arrays.equals(nGramRange, new Integer[]{1, 1})){
			DocumentBuilder documentBuilder = DOMUtil.createDocumentBuilder();

			InlineTable inlineTable = new InlineTable()
				.addRows(DOMUtil.createRow(documentBuilder, Arrays.asList("string", "stem", "regex"), Arrays.asList("(^|\\s+)\\p{Punct}*(" + JOINER.join(stopWords) + ")\\p{Punct}*(\\s+|$)", " ", "true")));

			TextIndexNormalization textIndexNormalization = new TextIndexNormalization()
				.setRecursive(Boolean.TRUE) // Handles consecutive matches. See http://stackoverflow.com/a/25085385
				.setInlineTable(inlineTable);

			textIndex.addTextIndexNormalizations(textIndexNormalization);
		}

		String name = functionName() + "@" + String.valueOf(CountVectorizer.SEQUENCE.getAndIncrement());

		DefineFunction defineFunction = new DefineFunction(name, OpType.CONTINUOUS, null)
			.setDataType(DataType.DOUBLE)
			.addParameterFields(documentField, termField)
			.setExpression(textIndex);

		return defineFunction;
	}

	public Apply encodeApply(String function, Feature feature, int index, String term){
		Constant constant = PMMLUtil.createConstant(term)
			.setDataType(DataType.STRING);

		return PMMLUtil.createApply(function, feature.ref(), constant);
	}

	public String functionName(){
		return "tf";
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

	public Object[] getNGramRange(){
		return (Object[])get("ngram_range");
	}

	public Object getPreprocessor(){
		return get("preprocessor");
	}

	public List<String> getStopWords(){
		Object stopWords = get("stop_words");

		if(stopWords instanceof String){
			return loadStopWords((String)stopWords);
		}

		return (List)stopWords;
	}

	public String getStripAccents(){
		return (String)get("strip_accents");
	}

	public Splitter getTokenizer(){
		Object tokenizer = get("tokenizer");

		try {
			if(tokenizer == null){
				throw new NullPointerException();
			}

			return (Splitter)tokenizer;
		} catch(RuntimeException re){
			throw new IllegalArgumentException("The tokenizer object (" + ClassDictUtil.formatClass(tokenizer) + ") is not Splitter");
		}
	}

	public String getTokenPattern(){
		return (String)get("token_pattern");
	}

	public Map<String, Scalar> getVocabulary(){
		return (Map)get("vocabulary_");
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

	private static final AtomicInteger SEQUENCE = new AtomicInteger(1);
}