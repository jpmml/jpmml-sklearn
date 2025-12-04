/*
 * Copyright (c) 2024 Villu Ruusmann
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

import org.jpmml.python.PythonException;

public class SkLearnException extends PythonException {

	public SkLearnException(String message){
		super(message);
	}

	public SkLearnException(String message, Throwable cause){
		super(message, cause);
	}

	public SkLearnException(String problem, String solution){
		super(formatMessage(problem, solution));
	}

	public SkLearnException(String problem, String solution, Throwable cause){
		super(formatMessage(problem, solution), cause);
	}

	static
	public String formatMessage(String problem, String solution){
		return (problem + ". " + solution);
	}
}