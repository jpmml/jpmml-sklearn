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
package sklearn2pmml.feature_extraction.text;

import java.util.List;

import org.dmg.pmml.TextIndex;
import sklearn.feature_extraction.text.Tokenizer;

public class Splitter extends Tokenizer {

	public Splitter(){
		this("sklearn2pmml.feature_extraction.text", "Splitter");
	}

	public Splitter(String module, String name){
		super(module, name);
	}

	@Override
	public TextIndex configure(TextIndex textIndex){
		String wordSeparatorRE = getWordSeparatorRE();

		return textIndex
			.setTokenize(Boolean.TRUE)
			.setWordSeparatorCharacterRE(wordSeparatorRE);
	}

	@Override
	public String formatStopWordsRE(List<String> stopWords){
		String wordSeparatorRE = getWordSeparatorRE();

		return "(^|" + wordSeparatorRE + ")\\p{Punct}*(" + String.join("|", stopWords) + ")\\p{Punct}*(" + wordSeparatorRE + "|$)";
	}

	public void __setstate__(String wordSeparatorRE){
		setWordSeparatorRE(wordSeparatorRE);
	}

	public String getWordSeparatorRE(){
		return getString("word_separator_re");
	}

	public Splitter setWordSeparatorRE(String wordSeparatorRE){
		setattr("word_separator_re", wordSeparatorRE);

		return this;
	}
}