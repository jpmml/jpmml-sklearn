/*
 * Copyright (c) 2017 Villu Ruusmann
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

import org.dmg.pmml.DataType;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.Expression;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.OpType;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.Feature;
import org.jpmml.python.DataFrameScope;
import org.jpmml.python.ExpressionTranslator;
import org.jpmml.python.Scope;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn.StepUtil;
import sklearn.Transformer;

public class ExpressionTransformer extends Transformer {

	public ExpressionTransformer(String module, String name){
		super(module, name);
	}

	@Override
	public List<Feature> encodeFeatures(List<Feature> features, SkLearnEncoder encoder){
		Object dtype = getDType();
		String expr = getExpr();

		Scope scope = new DataFrameScope(FieldName.create("X"), features);

		Expression expression = ExpressionTranslator.translate(expr, scope);

		DataType dataType;

		if(dtype != null){
			dataType = StepUtil.getDataType(dtype);
		} else

		{
			if(ExpressionTranslator.isString(expression, scope)){
				dataType = DataType.STRING;
			} else

			{
				dataType = DataType.DOUBLE;
			}
		}

		OpType opType = StepUtil.getOpType(dataType);

		DerivedField derivedField = encoder.createDerivedField(createFieldName("eval", expr), opType, dataType, expression);

		return Collections.singletonList(new ContinuousFeature(encoder, derivedField));
	}

	public Object getDType(){
		Object dtype = get("dtype");

		if(dtype == null){
			return null;
		}

		return super.getDType(true);
	}

	public String getExpr(){

		// SkLearn2PMML 0.31.0
		if(containsKey("expr_")){
			return getString("expr_");
		} else

		// SkLearn2PMML 0.31.1+
		{
			return getString("expr");
		}
	}
}