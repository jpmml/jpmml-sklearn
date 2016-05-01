/*
 * Copyright (c) 2015 Villu Ruusmann
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
package sklearn;

import org.dmg.pmml.DataType;
import org.dmg.pmml.DefineFunction;
import org.dmg.pmml.Expression;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.FieldRef;
import org.dmg.pmml.OpType;
import org.dmg.pmml.ParameterField;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.converter.Schema;

public class EstimatorUtil {

	private EstimatorUtil(){
	}

	static
	public Schema createSegmentSchema(Schema schema){
		Schema result = new Schema(null, schema.getTargetCategories(), schema.getActiveFields());

		return result;
	}

	static
	public DefineFunction encodeLogitFunction(){
		return encodeLossFunction("logit", -1d);
	}

	static
	public DefineFunction encodeAdaBoostFunction(){
		return encodeLossFunction("adaboost", -2d);
	}

	static
	private DefineFunction encodeLossFunction(String function, double multiplier){
		FieldName name = FieldName.create("value");

		ParameterField parameterField = new ParameterField(name)
			.setDataType(DataType.DOUBLE)
			.setOpType(OpType.CONTINUOUS);

		// "1 / (1 + exp($multiplier * $name))"
		Expression expression = PMMLUtil.createApply("/", PMMLUtil.createConstant(1d), PMMLUtil.createApply("+", PMMLUtil.createConstant(1d), PMMLUtil.createApply("exp", PMMLUtil.createApply("*", PMMLUtil.createConstant(multiplier), new FieldRef(name)))));

		DefineFunction defineFunction = new DefineFunction(function, OpType.CONTINUOUS, null)
			.setDataType(DataType.DOUBLE)
			.setOpType(OpType.CONTINUOUS)
			.addParameterFields(parameterField)
			.setExpression(expression);

		return defineFunction;
	}
}