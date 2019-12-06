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
package numpy.core;

import org.dmg.pmml.Expression;
import org.dmg.pmml.PMMLFunctions;
import org.jpmml.converter.PMMLUtil;

public class UFuncUtil {

	private UFuncUtil(){
	}

	static
	public Expression encodeUFunc(UFunc ufunc, Expression expression){
		return encodeUFunc(ufunc.getModule(), ufunc.getName(), expression);
	}

	static
	public Expression encodeUFunc(String module, String name, Expression expression){

		switch(module){
			case "numpy":
			case "numpy.core":
				return encodeNumpyUFunc(name, expression);
			default:
				throw new IllegalArgumentException(module);
		}
	}

	static
	private Expression encodeNumpyUFunc(String name, Expression expression){

		switch(name){
			case "absolute":
				return PMMLUtil.createApply(PMMLFunctions.ABS, expression);
			case "acos":
				return PMMLUtil.createApply(PMMLFunctions.ACOS, expression);
			case "asin":
				return PMMLUtil.createApply(PMMLFunctions.ASIN, expression);
			case "atan":
				return PMMLUtil.createApply(PMMLFunctions.ATAN, expression);
			case "atan2":
				return PMMLUtil.createApply(PMMLFunctions.ATAN2, expression);
			case "ceil":
				return PMMLUtil.createApply(PMMLFunctions.CEIL, expression);
			case "cos":
				return PMMLUtil.createApply(PMMLFunctions.COS, expression);
			case "cosh":
				return PMMLUtil.createApply(PMMLFunctions.COSH, expression);
			case "exp":
				return PMMLUtil.createApply(PMMLFunctions.EXP, expression);
			case "expm1":
				return PMMLUtil.createApply(PMMLFunctions.EXPM1, expression);
			case "floor":
				return PMMLUtil.createApply(PMMLFunctions.FLOOR, expression);
			case "hypot":
				return PMMLUtil.createApply(PMMLFunctions.HYPOT, expression);
			case "log":
				return PMMLUtil.createApply(PMMLFunctions.LN, expression);
			case "log1p":
				return PMMLUtil.createApply(PMMLFunctions.LN1P, expression);
			case "log10":
				return PMMLUtil.createApply(PMMLFunctions.LOG10, expression);
			case "negative":
				return PMMLUtil.createApply(PMMLFunctions.MULTIPLY, PMMLUtil.createConstant(-1), expression);
			case "reciprocal":
				return PMMLUtil.createApply(PMMLFunctions.DIVIDE, PMMLUtil.createConstant(1), expression);
			case "rint":
				return PMMLUtil.createApply(PMMLFunctions.RINT, expression);
			case "sign":
				return PMMLUtil.createApply(PMMLFunctions.IF, PMMLUtil.createApply(PMMLFunctions.LESSTHAN, expression, PMMLUtil.createConstant(0)),
					PMMLUtil.createConstant(-1), // x < 0
					PMMLUtil.createApply(PMMLFunctions.IF, PMMLUtil.createApply(PMMLFunctions.GREATERTHAN, expression, PMMLUtil.createConstant(0)),
						PMMLUtil.createConstant(+1), // x > 0
						PMMLUtil.createConstant(0) // x == 0
					)
				);
			case "sin":
				return PMMLUtil.createApply(PMMLFunctions.SIN, expression);
			case "sinh":
				return PMMLUtil.createApply(PMMLFunctions.SINH, expression);
			case "sqrt":
				return PMMLUtil.createApply(PMMLFunctions.SQRT, expression);
			case "square":
				return PMMLUtil.createApply(PMMLFunctions.MULTIPLY, expression, expression);
			case "tan":
				return PMMLUtil.createApply(PMMLFunctions.TAN, expression);
			case "tanh":
				return PMMLUtil.createApply(PMMLFunctions.TANH, expression);
			default:
				throw new IllegalArgumentException(name);
		}
	}
}