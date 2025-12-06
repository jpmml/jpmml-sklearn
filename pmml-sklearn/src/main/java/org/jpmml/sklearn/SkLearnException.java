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

public class SkLearnException extends RuntimeException {

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

	@Override
	synchronized
	public SkLearnException initCause(Throwable cause){
		return (SkLearnException)super.initCause(cause);
	}

	@Override
	synchronized
	public SkLearnException fillInStackTrace(){
		return (SkLearnException)super.fillInStackTrace();
	}

	static
	public String formatMessage(String problem, String solution){

		if(solution != null && !solution.isEmpty()){
			return (problem + ". To fix, " + solution.substring(0, 1).toLowerCase() + solution.substring(1));
		}

		return problem;
	}
}