/*
 * Copyright (c) 2018 Villu Ruusmann
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

public class XMLUtil {

	private XMLUtil(){
	}

	static
	public String createTagName(String string){
		StringBuilder sb = new StringBuilder();

		for(int i = 0; i < string.length(); i++){
			char c = string.charAt(i);

			boolean valid = (i == 0 ? isTagNameStartChar(c) : isTagNameContinuationChar(c));
			if(valid){
				sb.append(c);
			} else

			{
				if(c == ' '){
					sb.append("_x0020_");
				} else

				{
					sb.append('_');
					sb.append('x');

					String hex = Integer.toHexString(c);
					for(int j = 0; j < (4 - hex.length()); j++){
						sb.append('0');
					}

					sb.append(hex);
					sb.append('_');
				}
			}
		}

		return sb.toString();
	}

	static
	private boolean isTagNameStartChar(char c){

		switch(c){
			case '_':
				return true;
			default:
				return Character.isLetter(c);
		}
	}

	static
	private boolean isTagNameContinuationChar(char c){

		switch(c){
			case '-':
			case '.':
			case '_':
				return true;
			default:
				return Character.isLetterOrDigit(c);
		}
	}
}