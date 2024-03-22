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
package scipy.interpolate;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import com.google.common.collect.Iterables;
import org.dmg.pmml.Apply;
import org.dmg.pmml.DataType;
import org.dmg.pmml.DefineFunction;
import org.dmg.pmml.Expression;
import org.dmg.pmml.FieldRef;
import org.dmg.pmml.OpType;
import org.dmg.pmml.PMMLFunctions;
import org.dmg.pmml.ParameterField;
import org.jpmml.converter.CMatrixUtil;
import org.jpmml.converter.ExpressionUtil;
import org.jpmml.converter.FieldNameUtil;
import org.jpmml.converter.ValueUtil;
import org.jpmml.sklearn.SkLearnEncoder;

public class BSplineUtil {

	private BSplineUtil(){
	}

	static
	public List<DefineFunction> createSplineFunction(BSpline bspline, SkLearnEncoder encoder){
		int k = bspline.getK();

		List<Number> c = bspline.getC();
		List<Number> t = bspline.getT();

		// XXX
		int[] cShape = bspline.getArrayShape("c");

		int n = (t.size() - k - 1);

		if(cShape.length == 1){

			for(int i = 0; i < n; i++){

				for(int j = k; j >= 0; j--){
					createBasisFunction(t, i, j, encoder);
				}
			}

			DefineFunction splineFunction = createSplineFunction(null, c, n, k, encoder);

			return Collections.singletonList(splineFunction);
		} else

		if(cShape.length == 2){
			int rows = cShape[0];
			int columns = cShape[1];

			for(int i = 0; i < (n + k); i++){

				for(int j = k; j >= 0; j--){
					createBasisFunction(t, i, j, encoder);
				}
			}

			List<DefineFunction> result = new ArrayList<>();

			for(int i = 0; i < rows; i++){
				DefineFunction splineFunction = createSplineFunction(i, CMatrixUtil.getRow(c, rows, columns, i), n, k, encoder);

				result.add(splineFunction);
			}

			return result;
		} else

		{
			throw new IllegalArgumentException();
		}
	}

	static
	private DefineFunction createSplineFunction(Integer row, List<Number> c, int n, int k, SkLearnEncoder encoder){
		ParameterField valueField = new ParameterField()
			.setName("x")
			.setOpType(OpType.CONTINUOUS)
			.setDataType(DataType.DOUBLE);

		List<Apply> expressions = new ArrayList<>();

		for(int i = 0; i < n; i++){
			Number coefficient = c.get(i);

			if(ValueUtil.isZero(coefficient)){
				continue;
			} // End if

			Apply apply = ExpressionUtil.createApply(formatBFunction(i, k), new FieldRef(valueField));

			if(!ValueUtil.isOne(coefficient)){
				apply = ExpressionUtil.createApply(PMMLFunctions.MULTIPLY, ExpressionUtil.createConstant(coefficient), apply);
			}

			expressions.add(apply);
		}

		Expression expression;

		if(expressions.size() == 1){
			expression = Iterables.getOnlyElement(expressions);
		} else

		if(expressions.size() >= 2){
			expression = ExpressionUtil.createApply(PMMLFunctions.SUM)
				.addExpressions(expressions.toArray(new Expression[expressions.size()]));
		} else

		{
			throw new IllegalArgumentException();
		}

		DefineFunction splineFunction = new DefineFunction(formatBSplineFunction(k) + (row != null ? ("[" + row + "]") : ""), OpType.CONTINUOUS, DataType.DOUBLE, null, expression)
			.addParameterFields(valueField);

		encoder.addDefineFunction(splineFunction);

		return splineFunction;
	}

	static
	private DefineFunction createBasisFunction(List<Number> t, int i, int k, SkLearnEncoder encoder){
		ParameterField valueField = new ParameterField()
			.setName("x")
			.setOpType(OpType.CONTINUOUS)
			.setDataType(DataType.DOUBLE);

		Expression expression;

		if(k == 0){

			if(!(t.get(i)).equals(t.get(i + 1))){
				expression = ExpressionUtil.createApply(PMMLFunctions.IF,
					ExpressionUtil.createApply(PMMLFunctions.AND,
						ExpressionUtil.createApply(PMMLFunctions.GREATEROREQUAL, new FieldRef(valueField), ExpressionUtil.createConstant(t.get(i))),
						ExpressionUtil.createApply(PMMLFunctions.LESSTHAN, new FieldRef(valueField), ExpressionUtil.createConstant(t.get(i + 1)))
					),
					ExpressionUtil.createConstant(1d),
					ExpressionUtil.createConstant(0d)
				);
			} else

			{
				expression = ExpressionUtil.createConstant(0d);
			}
		} else

		{
			List<Apply> expressions = new ArrayList<>(2);

			if((i + k) < t.size() && !(t.get(i + k)).equals(t.get(i))){
				Apply apply = ExpressionUtil.createApply(PMMLFunctions.DIVIDE,
					ExpressionUtil.createApply(PMMLFunctions.SUBTRACT, new FieldRef(valueField), ExpressionUtil.createConstant(t.get(i))),
					ExpressionUtil.createConstant((t.get(i + k)).doubleValue() - (t.get(i)).doubleValue())
				);

				apply = ExpressionUtil.createApply(PMMLFunctions.MULTIPLY, apply, ExpressionUtil.createApply(formatBFunction(i, k - 1), new FieldRef(valueField)));

				expressions.add(apply);
			} // End if

			if((i + k + 1) < t.size() && !(t.get(i + k + 1)).equals(t.get(i + 1))){
				Apply apply = ExpressionUtil.createApply(PMMLFunctions.DIVIDE,
					ExpressionUtil.createApply(PMMLFunctions.SUBTRACT, ExpressionUtil.createConstant(t.get(i + k + 1)), new FieldRef(valueField)),
					ExpressionUtil.createConstant((t.get(i + k + 1)).doubleValue() - (t.get(i + 1)).doubleValue())
				);

				apply = ExpressionUtil.createApply(PMMLFunctions.MULTIPLY, apply, ExpressionUtil.createApply(formatBFunction(i + 1, k - 1), new FieldRef(valueField)));

				expressions.add(apply);
			} // End if

			if(expressions.size() == 2){
				expression = ExpressionUtil.createApply(PMMLFunctions.ADD, expressions.get(0), expressions.get(1));
			} else

			if(expressions.size() == 1){
				expression = Iterables.getOnlyElement(expressions);
			} else

			{
				expression = ExpressionUtil.createConstant(0d);
			}
		}

		DefineFunction defineFunction = new DefineFunction(formatBFunction(i, k), OpType.CONTINUOUS, DataType.DOUBLE, null, expression)
			.addParameterFields(valueField);

		encoder.addDefineFunction(defineFunction);

		return defineFunction;
	}

	static
	private String formatBSplineFunction(int k){
		return FieldNameUtil.create("scipy.interpolate.BSpline", k);
	}

	static
	private String formatBFunction(int i, int k){
		return FieldNameUtil.create("scipy.interpolate.B", i, k);
	}
}