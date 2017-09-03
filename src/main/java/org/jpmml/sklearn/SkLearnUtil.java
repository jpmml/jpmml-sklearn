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
package org.jpmml.sklearn;

import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class SkLearnUtil {

	private SkLearnUtil(){
	}

	static
	public int compareVersion(String left, String right){
		List<Integer> leftTokens = parseVersion(left);
		List<Integer> rightTokens = parseVersion(right);

		for(int i = 0; i < Math.min(leftTokens.size(), rightTokens.size()); i++){
			int diff = Integer.compare(leftTokens.get(i), rightTokens.get(i));

			if(diff != 0){
				return diff;
			}
		}

		if((leftTokens.size() < rightTokens.size()) && rightTokens.get(leftTokens.size()) != 0){
			return -1;
		}

		return 0;
	}

	static
	public List<Integer> parseVersion(String string){
		Matcher matcher = PEP440_VERSION.matcher(string);

		if(!matcher.matches()){
			throw new IllegalArgumentException(string);
		}

		List<Integer> tokens = new ArrayList<>();

		for(int i = 1; i <= 3; i++){
			String token = matcher.group(i);

			if(token == null){
				break;
			}

			tokens.add(new Integer(token));
		}

		return tokens;
	}

	private static final Pattern PEP440_VERSION = Pattern.compile("(\\d+)\\.(\\d+)(?:(?:a|b|rc)\\d)?(?:\\.(?:(\\d)|(?:(?:dev)?\\d?)))?");
}