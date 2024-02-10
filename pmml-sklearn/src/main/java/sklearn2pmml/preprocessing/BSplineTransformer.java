/*
 * Copyright (c) 2020 Villu Ruusmann
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

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import com.google.common.collect.Iterables;
import org.dmg.pmml.Apply;
import org.dmg.pmml.DataType;
import org.dmg.pmml.DefineFunction;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.Expression;
import org.dmg.pmml.FieldRef;
import org.dmg.pmml.OpType;
import org.dmg.pmml.PMMLFunctions;
import org.dmg.pmml.ParameterField;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.ExpressionUtil;
import org.jpmml.converter.Feature;
import org.jpmml.python.ClassDictUtil;
import org.jpmml.sklearn.SkLearnEncoder;
import scipy.interpolate.BSpline;
import sklearn.Transformer;

public class BSplineTransformer extends Transformer {

	public BSplineTransformer(String module, String name){
		super(module, name);
	}

	@Override
	public List<Feature> encodeFeatures(List<Feature> features, SkLearnEncoder encoder){
		BSpline bspline = getBSpline();

		ClassDictUtil.checkSize(1, features);

		Feature feature = features.get(0);

		ContinuousFeature continuousFeature = feature.toContinuousFeature();

		DefineFunction defineFunction = createBSplineFunction(bspline, encoder);

		Apply apply = ExpressionUtil.createApply(defineFunction, continuousFeature.ref());

		DerivedField derivedField = encoder.createDerivedField(createFieldName("bspline", continuousFeature), apply);

		return Collections.singletonList(new ContinuousFeature(encoder, derivedField));
	}

	public BSpline getBSpline(){
		return get("bspline", BSpline.class);
	}

	/**
	 * https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BSpline.html
	 */
	static
	private DefineFunction createBSplineFunction(BSpline bspline, SkLearnEncoder encoder){
		int k = bspline.getK();

		List<Number> c = bspline.getC();
		List<Number> t = bspline.getT();

		int n = (t.size() - k - 1);

		ParameterField valueField = new ParameterField()
			.setName("x")
			.setOpType(OpType.CONTINUOUS)
			.setDataType(DataType.DOUBLE);

		Apply sumApply = ExpressionUtil.createApply(PMMLFunctions.SUM);

		for(int i = 0; i < n; i++){

			for(int j = k; j >= 0; j--){
				createBFunction(t, i, j, encoder);
			}

			Apply apply = ExpressionUtil.createApply(PMMLFunctions.MULTIPLY, ExpressionUtil.createConstant(c.get(i)), ExpressionUtil.createApply(formatBFunction(i, k), new FieldRef(valueField)));

			sumApply.addExpressions(apply);
		}

		DefineFunction defineFunction = new DefineFunction(formatBSplineFunction(k), OpType.CONTINUOUS, DataType.DOUBLE, null, sumApply)
			.addParameterFields(valueField);

		encoder.addDefineFunction(defineFunction);

		return defineFunction;
	}

	static
	private DefineFunction createBFunction(List<Number> t, int i, int k, SkLearnEncoder encoder){
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

			if(!(t.get(i + k)).equals(t.get(i))){
				Apply apply = ExpressionUtil.createApply(PMMLFunctions.DIVIDE,
					ExpressionUtil.createApply(PMMLFunctions.SUBTRACT, new FieldRef(valueField), ExpressionUtil.createConstant(t.get(i))),
					ExpressionUtil.createConstant((t.get(i + k)).doubleValue() - (t.get(i)).doubleValue())
				);

				apply = ExpressionUtil.createApply(PMMLFunctions.MULTIPLY, apply, ExpressionUtil.createApply(formatBFunction(i, k - 1), new FieldRef(valueField)));

				expressions.add(apply);
			} // End if

			if(!(t.get(i + k + 1)).equals(t.get(i + 1))){
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
		return "scipy.interpolate.BSpline(" + k + ")";
	}

	static
	private String formatBFunction(int i, int k){
		return "scipy.interpolate.B(" + i + ", " + k + ")";
	}
}