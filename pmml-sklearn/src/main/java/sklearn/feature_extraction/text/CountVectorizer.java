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
import org.jpmml.converter.CategoricalFeature;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.ExpressionUtil;
import org.jpmml.converter.Feature;
import org.jpmml.converter.FieldNameUtil;
import org.jpmml.converter.ObjectFeature;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.converter.StringFeature;
import org.jpmml.converter.TypeUtil;
import org.jpmml.converter.ValueUtil;
import org.jpmml.python.AttributeException;
import org.jpmml.python.ClassDictUtil;
import org.jpmml.python.TypeInfo;
import org.jpmml.sklearn.SkLearnEncoder;
import org.jpmml.sklearn.SkLearnException;
import sklearn.HasSparseOutput;
import sklearn.SkLearnTransformer;
import sklearn2pmml.feature_extraction.text.Matcher;
import sklearn2pmml.feature_extraction.text.Splitter;

public class CountVectorizer extends SkLearnTransformer implements HasSparseOutput {

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
		Boolean binary = getBinary();
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
		OpType opType = TypeUtil.getOpType(dataType);

		if(binary){
			opType = OpType.CATEGORICAL;
		} // End if

		if(lowercase){
			Apply apply = ExpressionUtil.createApply(PMMLFunctions.LOWERCASE, feature.ref());

			DerivedField derivedField = encoder.ensureDerivedField(FieldNameUtil.create(PMMLFunctions.LOWERCASE, feature), OpType.CATEGORICAL, DataType.STRING, () -> apply);

			feature = new StringFeature(encoder, derivedField);
		}

		DefineFunction defineFunction = encodeDefineFunction(feature, encoder);

		encoder.addDefineFunction(defineFunction);

		List<Feature> result = new ArrayList<>();

		for(int i = 0, max = indexTermMap.size(); i < max; i++){
			String term = indexTermMap.get(i);

			Apply apply = encodeApply(defineFunction, feature, i, term);

			DerivedField derivedField = encoder.createDerivedField(FieldNameUtil.create(functionName(), feature, term), opType, dataType, apply);

			Feature termFeature;

			if(binary){
				termFeature = new CategoricalFeature(encoder, derivedField, Arrays.asList(0, 1));
			} else

			{
				termFeature = new ObjectFeature(encoder, derivedField){

					@Override
					public ContinuousFeature toContinuousFeature(){
						return new ContinuousFeature(encoder, this);
					}
				};
			}

			result.add(termFeature);
		}

		return result;
	}

	public DefineFunction encodeDefineFunction(Feature feature, SkLearnEncoder encoder){
		@SuppressWarnings("unused")
		String analyzer = getAnalyzer();
		List<String> stopWords = getStopWords();
		Object[] nGramRange = getNGramRange();
		Boolean binary = getBinary();
		@SuppressWarnings("unused")
		Object preprocessor = getPreprocessor();
		@SuppressWarnings("unused")
		String stripAccents = getStripAccents();
		Tokenizer tokenizer = getTokenizer();

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
		Constant constant = ExpressionUtil.createConstant(DataType.STRING, term);

		return ExpressionUtil.createApply(defineFunction, feature.ref(), constant);
	}

	public String functionName(){
		return "tf";
	}

	public String getAnalyzer(){
		return getEnum("analyzer", this::getString, Arrays.asList(CountVectorizer.ANALYZER_WORD));
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
		Object object = getOptionalObject("preprocessor");

		if(object != null){
			throw new AttributeException("Attribute \'" + ClassDictUtil.formatMember(this, "preprocessor") + "\' must be set to the missing (None) value");
		}

		return object;
	}

	@Override
	public Boolean getSparseOutput(){
		return Boolean.TRUE;
	}

	public List<String> getStopWords(){
		Object stopWords = getOptionalObject("stop_words");

		if(stopWords instanceof String){
			return loadStopWords((String)stopWords);
		}

		return (List)stopWords;
	}

	public String getStripAccents(){
		return getOptionalEnum("strip_accents", this::getOptionalString, Collections.emptyList());
	}

	public Tokenizer getTokenizer(){
		Object object = getOptionalObject("tokenizer");

		if((object != null) && !(Tokenizer.class).isInstance(object)){
			throw new AttributeException("Attribute \'" + ClassDictUtil.formatMember(this, "tokenizer") + "\' has an unsupported value (" + ClassDictUtil.formatClass(object) +"). Supported Python classes are " + Arrays.asList(Matcher.class.getName(), Splitter.class.getName()));
		}

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
			throw new SkLearnException("Failed to locate the stop words list resource \'" + stopWords + "\'");
		}

		try(Reader reader = new InputStreamReader(is, "UTF-8")){
			return CharStreams.readLines(reader);
		} catch(Exception e){
			throw new SkLearnException("Failed to load the stop words list", e);
		}
	}

	public static final String TOKEN_PATTERN = "(?u)\\b\\w\\w+\\b";

	private static final String ANALYZER_WORD = "word";

}
