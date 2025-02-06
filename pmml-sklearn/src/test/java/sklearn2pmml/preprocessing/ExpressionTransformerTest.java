/*
 * Copyright (c) 2022 Villu Ruusmann
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

import java.util.Collections;
import java.util.List;

import com.google.common.collect.Iterables;
import org.dmg.pmml.Apply;
import org.dmg.pmml.DataField;
import org.dmg.pmml.DataType;
import org.dmg.pmml.Expression;
import org.dmg.pmml.Field;
import org.dmg.pmml.FieldRef;
import org.dmg.pmml.HasExpression;
import org.dmg.pmml.InvalidValueTreatmentMethod;
import org.dmg.pmml.OpType;
import org.dmg.pmml.PMMLFunctions;
import org.jpmml.converter.ExpressionUtil;
import org.jpmml.converter.Feature;
import org.jpmml.converter.WildcardFeature;
import org.jpmml.model.ReflectionUtil;
import org.jpmml.sklearn.SkLearnEncoder;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

public class ExpressionTransformerTest {

	@Test
	public void encode(){
		String expr = "X[0]";

		assertNull(encode(expr, null, null, null));

		Expression expected = new FieldRef("x");

		expected = ((FieldRef)expected)
			.setMapMissingTo(-1d);

		assertTrue(ReflectionUtil.equals(expected, encode(expr, -1d, null, null)));

		try {
			encode(expr, null, -1d, null);

			fail();
		} catch(ClassCastException cce){
			// Ignored
		}

		expr = "1.0";

		expected = ExpressionUtil.createConstant(DataType.DOUBLE, 1d);

		try {
			encode(expr, null, null, "as_missing");

			fail();
		} catch(ClassCastException cce){
			// Ignored
		}

		expr = "X[0] + 1.0";

		expected = new Apply(PMMLFunctions.ADD)
			.addExpressions(new FieldRef("x"), ExpressionUtil.createConstant(DataType.DOUBLE, 1d));

		expected = ((Apply)expected)
			.setMapMissingTo(-1d)
			.setDefaultValue(null)
			.setInvalidValueTreatment(InvalidValueTreatmentMethod.RETURN_INVALID);

		assertTrue(ReflectionUtil.equals(expected, encode(expr, -1d, Double.NaN, "return_invalid")));

		expected = ((Apply)expected)
			.setMapMissingTo(null)
			.setDefaultValue(-1d)
			.setInvalidValueTreatment(InvalidValueTreatmentMethod.AS_MISSING);

		assertTrue(ReflectionUtil.equals(expected, encode(expr, Double.NaN, -1d, "as_missing")));
	}

	static
	private Expression encode(String expr, Object mapMissingTo, Object defaultValue, String invalidValueTreatment){
		ExpressionTransformer expressionTransformer = new ExpressionTransformer();

		expressionTransformer
			.setExpr(expr)
			.setMapMissingTo(mapMissingTo)
			.setDefaultValue(defaultValue)
			.setInvalidValueTreatment(invalidValueTreatment);

		return encode(expressionTransformer);
	}

	static
	private Expression encode(ExpressionTransformer expressionTransformer){
		SkLearnEncoder encoder = new SkLearnEncoder();

		DataField dataField = encoder.createDataField("x", OpType.CONTINUOUS, DataType.DOUBLE);

		Feature inputFeature = new WildcardFeature(encoder, dataField);

		List<Feature> outputFeatures = expressionTransformer.encode(Collections.singletonList(inputFeature), encoder);

		Feature outputFeature = Iterables.getOnlyElement(outputFeatures);

		Field<?> field = outputFeature.getField();

		if(field instanceof HasExpression){
			HasExpression<?> hasExpression = (HasExpression<?>)field;

			return hasExpression.getExpression();
		}

		return null;
	}
}