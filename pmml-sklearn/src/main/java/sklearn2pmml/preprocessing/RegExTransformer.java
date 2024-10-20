/*
 * Copyright (c) 2019 Villu Ruusmann
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
package sklearn2pmml.preprocessing;

import java.util.Arrays;

import org.dmg.pmml.DataType;
import org.dmg.pmml.OpType;
import sklearn.Transformer;

abstract
public class RegExTransformer extends Transformer {

	public RegExTransformer(String module, String name){
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

	public String getPattern(){
		return getString("pattern");
	}

	RegExTransformer setPattern(String pattern){
		setattr("pattern", pattern);

		return this;
	}

	public String getReFlavour(){
		return getOptionalEnum("re_flavour", this::getOptionalString, Arrays.asList(RegExTransformer.RE_FLAVOUR_PCRE, RegExTransformer.RE_FLAVOUR_PCRE2, RegExTransformer.RE_FLAVOUR_RE));
	}

	RegExTransformer setReFlavour(String reFlavour){
		setattr("re_flavour", reFlavour);

		return this;
	}

	static
	public String translatePattern(String pattern, String reFlavour){

		switch(reFlavour){
			case RegExTransformer.RE_FLAVOUR_PCRE:
			case RegExTransformer.RE_FLAVOUR_PCRE2:
			case RegExTransformer.RE_FLAVOUR_RE:
				return pattern;
			default:
				throw new IllegalArgumentException(reFlavour);
		}
	}

	static
	protected String formatArg(String string){
		return ("\"" + string + "\"");
	}

	protected static final String RE_FLAVOUR_PCRE = "pcre";
	protected static final String RE_FLAVOUR_PCRE2 = "pcre2";
	protected static final String RE_FLAVOUR_RE = "re";
}