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
package sklearn;

import java.util.Arrays;
import java.util.List;

import org.dmg.pmml.DataType;
import org.dmg.pmml.Expression;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.FieldRef;
import org.dmg.pmml.OpType;
import org.dmg.pmml.Output;
import org.dmg.pmml.OutputField;
import org.dmg.pmml.PMMLFunctions;
import org.jpmml.converter.FieldNameUtil;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.converter.Transformation;

public class SkLearnOutlierTransformation implements Transformation {

	@Override
	public FieldName getName(FieldName name){
		return FieldNameUtil.create(Estimator.FIELD_PREDICT, name);
	}

	@Override
	public OpType getOpType(OpType opType){
		return OpType.CATEGORICAL;
	}

	@Override
	public DataType getDataType(DataType dataType){
		return DataType.INTEGER;
	}

	@Override
	public boolean isFinalResult(){
		return true;
	}

	@Override
	public Expression createExpression(FieldRef fieldRef){
		return PMMLUtil.createApply(PMMLFunctions.IF, fieldRef, PMMLUtil.createConstant(VALUE_OUTLIER), PMMLUtil.createConstant(VALUE_INLIER));
	}

	static
	public void decorate(Output output){

		if(output != null && output.hasOutputFields()){
			List<OutputField> outputFields = output.getOutputFields();

			OutputField finalOutputField = outputFields.get(outputFields.size() - 1);

			PMMLUtil.addValues(finalOutputField, Arrays.asList(VALUE_OUTLIER, VALUE_INLIER));
		}
	}

	public static final Integer VALUE_INLIER = +1;
	public static final Integer VALUE_OUTLIER = -1;
}