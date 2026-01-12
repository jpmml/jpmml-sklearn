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
package org.jpmml.sklearn;

import java.util.logging.LogRecord;
import java.util.logging.SimpleFormatter;
import java.util.regex.Pattern;

public class SkLearnFormatter extends SimpleFormatter {

	@Override
	public String format(LogRecord record){
		String string = super.format(record);

		Throwable throwable = record.getThrown();
		if(throwable != null){
			String throwableString = throwable.toString();

			string = string.replaceFirst("(?m)^" + Pattern.quote(throwableString), SkLearnFormatter.EXCEPTION_HEADER + throwableString);
			string = string.replaceAll("(?m)^\\s*Caused by:\\s*", SkLearnFormatter.CAUSED_BY_HEADER);
		}

		return string;
	}

	static
	private String center(String text){
		int spaces = (HEADER_WIDTH - text.length()) / 2;

		return (" ").repeat(spaces) + text;
	}

	private static final int HEADER_WIDTH = 64;

	private static final String THICK_LINE = ("=").repeat(HEADER_WIDTH);
	private static final String THIN_LINE = ("-").repeat(HEADER_WIDTH);

	public static String EXCEPTION_HEADER = "\n" + THICK_LINE + "\n" + center("EXCEPTION") + "\n" + THICK_LINE + "\n";
	public static String CAUSED_BY_HEADER = "\n" + THIN_LINE + "\n" + center("Caused by") + "\n" + THIN_LINE + "\n";
}