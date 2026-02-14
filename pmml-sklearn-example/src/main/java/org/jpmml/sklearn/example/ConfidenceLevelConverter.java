/*
 * Copyright (c) 2026 Villu Ruusmann
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
package org.jpmml.sklearn.example;

import com.beust.jcommander.IStringConverter;
import com.beust.jcommander.converters.BooleanConverter;
import com.beust.jcommander.converters.DoubleConverter;

/**
 * @see BooleanConverter
 * @see DoubleConverter
 */
public class ConfidenceLevelConverter implements IStringConverter<ConfidenceLevel> {

	@Override
	public ConfidenceLevel convert(String string){

		if(("false").equalsIgnoreCase(string) || ("true").equalsIgnoreCase(string)){
			return new ConfidenceLevel(Boolean.valueOf(string));
		}

		try {
			return new ConfidenceLevel(Double.valueOf(string));
		} catch(NumberFormatException nfe){
			// Ignored
		}

		return new ConfidenceLevel(string);
	}
}