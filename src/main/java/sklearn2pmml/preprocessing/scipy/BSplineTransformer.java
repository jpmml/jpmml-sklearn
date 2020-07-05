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
package sklearn2pmml.preprocessing.scipy;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import com.google.common.collect.Iterables;
import org.dmg.pmml.Apply;
import org.dmg.pmml.DataType;
import org.dmg.pmml.DefineFunction;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.Expression;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.FieldRef;
import org.dmg.pmml.OpType;
import org.dmg.pmml.PMMLFunctions;
import org.dmg.pmml.ParameterField;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.Feature;
import org.jpmml.converter.PMMLUtil;
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

		Apply apply = PMMLUtil.createApply(defineFunction.getName())
			.addExpressions(continuousFeature.ref());

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

		ParameterField paramterField = new ParameterField()
			.setName(FieldName.create("x"))
			.setOpType(OpType.CONTINUOUS)
			.setDataType(DataType.DOUBLE);

		Apply sumApply = PMMLUtil.createApply(PMMLFunctions.SUM);

		for(int i = 0; i < n; i++){

			for(int j = k; j >= 0; j--){
				createBFunction(t, i, j, encoder);
			}

			Apply apply = PMMLUtil.createApply(PMMLFunctions.MULTIPLY)
				.addExpressions(PMMLUtil.createConstant(c.get(i)))
				.addExpressions(PMMLUtil.createApply(formatBFunction(i, k), new FieldRef(paramterField.getName())));

			sumApply.addExpressions(apply);
		}

		DefineFunction defineFunction = new DefineFunction(formatBSplineFunction(k), OpType.CONTINUOUS, DataType.DOUBLE, null, sumApply)
			.addParameterFields(paramterField);

		encoder.addDefineFunction(defineFunction);

		return defineFunction;
	}

	static
	private DefineFunction createBFunction(List<Number> t, int i, int k, SkLearnEncoder encoder){
		ParameterField parameterField = new ParameterField()
			.setName(FieldName.create("x"))
			.setOpType(OpType.CONTINUOUS)
			.setDataType(DataType.DOUBLE);

		Expression expression;

		if(k == 0){

			if(!(t.get(i)).equals(t.get(i + 1))){
				expression = PMMLUtil.createApply(PMMLFunctions.IF)
					.addExpressions(PMMLUtil.createApply(PMMLFunctions.AND)
						.addExpressions(PMMLUtil.createApply(PMMLFunctions.GREATEROREQUAL, new FieldRef(parameterField.getName()), PMMLUtil.createConstant(t.get(i))))
						.addExpressions(PMMLUtil.createApply(PMMLFunctions.LESSTHAN, new FieldRef(parameterField.getName()), PMMLUtil.createConstant(t.get(i + 1))))
					)
					.addExpressions(PMMLUtil.createConstant(1d), PMMLUtil.createConstant(0d));
			} else

			{
				expression = PMMLUtil.createConstant(0d);
			}
		} else

		{
			List<Apply> expressions = new ArrayList<>(2);

			if(!(t.get(i + k)).equals(t.get(i))){
				Apply apply = PMMLUtil.createApply(PMMLFunctions.DIVIDE)
					.addExpressions(PMMLUtil.createApply(PMMLFunctions.SUBTRACT, new FieldRef(parameterField.getName()), PMMLUtil.createConstant(t.get(i))))
					.addExpressions(PMMLUtil.createConstant((t.get(i + k)).doubleValue() - (t.get(i)).doubleValue()));

				apply = PMMLUtil.createApply(PMMLFunctions.MULTIPLY, apply)
					.addExpressions(PMMLUtil.createApply(formatBFunction(i, k - 1), new FieldRef(parameterField.getName())));

				expressions.add(apply);
			} // End if

			if(!(t.get(i + k + 1)).equals(t.get(i + 1))){
				Apply apply = PMMLUtil.createApply(PMMLFunctions.DIVIDE)
					.addExpressions(PMMLUtil.createApply(PMMLFunctions.SUBTRACT, PMMLUtil.createConstant(t.get(i + k + 1)), new FieldRef(parameterField.getName())))
					.addExpressions(PMMLUtil.createConstant((t.get(i + k + 1)).doubleValue() - (t.get(i + 1)).doubleValue()));

				apply = PMMLUtil.createApply(PMMLFunctions.MULTIPLY, apply)
					.addExpressions(PMMLUtil.createApply(formatBFunction(i + 1, k - 1), new FieldRef(parameterField.getName())));

				expressions.add(apply);
			} // End if

			if(expressions.size() == 2){
				expression = PMMLUtil.createApply(PMMLFunctions.ADD)
					.addExpressions(expressions.get(0), expressions.get(1));
			} else

			if(expressions.size() == 1){
				expression = Iterables.getOnlyElement(expressions);
			} else

			{
				expression = PMMLUtil.createConstant(0d);
			}
		}

		DefineFunction defineFunction = new DefineFunction(formatBFunction(i, k), OpType.CONTINUOUS, DataType.DOUBLE, null, expression)
			.addParameterFields(parameterField);

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