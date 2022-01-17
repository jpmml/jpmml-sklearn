/*
 * Copyright (c) 2021 Villu Ruusmann
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
package org.jpmml.sklearn;

import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import com.google.common.base.Equivalence;
import com.google.common.collect.MapDifference;
import com.google.common.collect.Maps;
import org.dmg.pmml.TextIndex;
import org.dmg.pmml.TextIndexNormalization;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.evaluator.TextUtil;
import org.jpmml.evaluator.TokenizedString;
import org.jpmml.evaluator.testing.Batch;
import org.jpmml.evaluator.testing.Conflict;
import org.junit.Test;
import sklearn.feature_extraction.text.CountVectorizer;
import sklearn.feature_extraction.text.Tokenizer;
import sklearn2pmml.feature_extraction.text.Matcher;
import sklearn2pmml.feature_extraction.text.Splitter;

import static org.junit.Assert.fail;

public class TokenizerTest extends SkLearnTest implements SkLearnDatasets {

	@Test
	public void split() throws Exception {
		Splitter splitter = new Splitter()
			.setWordSeparatorRE("\\s+");

		evaluate("Splitter", SENTIMENT, splitter);
	}

	@Test
	public void match() throws Exception {
		Matcher matcher = new Matcher()
			.setWordRE(CountVectorizer.TOKEN_PATTERN);

		evaluate("CountVectorizer", SENTIMENT, matcher);

		matcher = new Matcher()
			.setWordRE("\\w+");

		evaluate("Matcher", SENTIMENT, matcher);
	}

	private void evaluate(String algorithm, String dataset, Tokenizer tokenizer) throws Exception {
		Batch batch = new SkLearnTestBatch(algorithm, dataset, (x) -> true, Equivalence.equals()){

			@Override
			public SkLearnTest getArchiveBatchTest(){
				return TokenizerTest.this;
			}
		};

		List<? extends Map<String, ?>> input = batch.getInput();
		List<? extends Map<String, ?>> output = batch.getOutput();

		if(input.size() != output.size()){
			throw new IllegalArgumentException();
		}

		Equivalence<Object> equivalence = batch.getEquivalence();

		// See src/test/resources/extensions/text.py
		List<String> stopWords = Arrays.asList("a", "and", "are", "d", "i", "is", "it", "ll", "m", "s", "the", "ve", "we", "you");

		TextIndex textIndex = createTextIndex(tokenizer, stopWords);

		boolean success = true;

		for(int i = 0; i < input.size(); i++){
			String inputSentence = getSentence(input.get(i));
			String outputSentence = getSentence(output.get(i));

			TokenizedString actualTokens = tokenize(textIndex, inputSentence.toLowerCase());
			Map<String, TokenizedString> actualResults = Collections.singletonMap("Sentence", actualTokens);

			TokenizedString expectedTokens = new TokenizedString(outputSentence.split("\t"));
			Map<String, TokenizedString> expectedResults = Collections.singletonMap("Sentence", expectedTokens);

			MapDifference<String, ?> difference = Maps.<String, Object>difference(expectedResults, actualResults, equivalence);
			if(!difference.areEqual()){
				Conflict conflict = new Conflict(i, Collections.emptyMap(), difference);

				System.err.println(conflict);

				success = false;
			}
		}

		if(!success){
			fail();
		}
	}

	static
	private String getSentence(Map<String, ?> map){
		return (String)map.get("Sentence");
	}

	static
	private TokenizedString tokenize(TextIndex textIndex, String string){
		string = TextUtil.normalize(textIndex, string);

		return TextUtil.tokenize(textIndex, string);
	}

	static
	private TextIndex createTextIndex(Tokenizer tokenizer, List<String> stopWords){
		TextIndex textIndex = new TextIndex();

		textIndex = tokenizer.configure(textIndex);

		Map<String, List<String>> data = new LinkedHashMap<>();
		data.put("string", Collections.singletonList(tokenizer.formatStopWordsRE(stopWords)));
		data.put("stem", Collections.singletonList(" "));
		data.put("regex", Collections.singletonList("true"));

		TextIndexNormalization textIndexNormalization = new TextIndexNormalization(PMMLUtil.createInlineTable(data))
			.setRecursive(Boolean.TRUE);

		textIndex.addTextIndexNormalizations(textIndexNormalization);

		return textIndex;
	}
}