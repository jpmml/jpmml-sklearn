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
package sklearn2pmml.preprocessing;

import java.util.Collections;
import java.util.List;

import com.google.common.collect.Iterables;
import org.dmg.pmml.Apply;
import org.dmg.pmml.DataField;
import org.dmg.pmml.DataType;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.Expression;
import org.dmg.pmml.OpType;
import org.jpmml.converter.Feature;
import org.jpmml.converter.WildcardFeature;
import org.jpmml.evaluator.EvaluationContext;
import org.jpmml.evaluator.ExpressionUtil;
import org.jpmml.evaluator.FieldValue;
import org.jpmml.evaluator.FieldValueUtil;
import org.jpmml.evaluator.VirtualEvaluationContext;
import org.jpmml.sklearn.SkLearnEncoder;

abstract
public class RegExTransformerTest {

	static
	Object evaluate(Expression expression, String string){
		EvaluationContext content = new VirtualEvaluationContext();
		content.declare("x", string);

		FieldValue value = ExpressionUtil.evaluate(expression, content);

		return FieldValueUtil.getValue(value);
	}

	static
	Apply encode(RegExTransformer regExTransformer){
		SkLearnEncoder encoder = new SkLearnEncoder();

		DataField dataField = encoder.createDataField("x", OpType.CATEGORICAL, DataType.STRING);

		Feature inputFeature = new WildcardFeature(encoder, dataField);

		List<Feature> outputFeatures = regExTransformer.encode(Collections.singletonList(inputFeature), encoder);

		Feature outputFeature = Iterables.getOnlyElement(outputFeatures);

		DerivedField derivedField = (DerivedField)outputFeature.getField();

		return (Apply)derivedField.getExpression();
	}
}