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
package sklearn2pmml.feature_extraction.text;

import java.util.List;

import com.google.common.base.Joiner;
import org.dmg.pmml.TextIndex;

public class Matcher extends Tokenizer {

	public Matcher(){
		this("sklearn2pmml.feature_extraction.text", "Matcher");
	}

	public Matcher(String module, String name){
		super(module, name);
	}

	@Override
	public TextIndex configure(TextIndex textIndex){
		String wordRE = getWordRE();

		return textIndex
			.setTokenize(Boolean.TRUE)
			.setWordRE(wordRE);
	}

	@Override
	public String formatStopWordsRE(List<String> stopWords){
		String wordRE = getWordRE();

		boolean unicode = wordRE.startsWith("(?u)");

		Joiner joiner = Joiner.on("|");

		return (unicode ? "(?u)" : "") + "\\b(" + joiner.join(stopWords) + ")\\b";
	}

	public void __setstate__(String wordRE){
		setWordRE(wordRE);
	}

	public String getWordRE(){
		return getString("word_re");
	}

	public Matcher setWordRE(String wordRE){
		put("word_re", wordRE);

		return this;
	}
}