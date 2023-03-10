/*
 * Copyright (c) 2023 Villu Ruusmann
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
package sklearn2pmml.expression;

import java.util.Collections;
import java.util.List;

import org.dmg.pmml.DataType;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.OpType;
import org.dmg.pmml.regression.RegressionModel;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.Feature;
import org.jpmml.converter.FieldNameUtil;
import org.jpmml.converter.Label;
import org.jpmml.converter.PMMLEncoder;
import org.jpmml.converter.Schema;
import org.jpmml.converter.regression.RegressionModelUtil;
import org.jpmml.python.DataFrameScope;
import org.jpmml.python.Scope;
import sklearn.Regressor;
import sklearn2pmml.util.EvaluatableUtil;
import sklearn2pmml.util.Expression;

public class ExpressionRegressor extends Regressor {

	public ExpressionRegressor(String module, String name){
		super(module, name);
	}

	@Override
	public RegressionModel encodeModel(Schema schema){
		Expression expr = getExpr();

		PMMLEncoder encoder = schema.getEncoder();

		Label label = schema.getLabel();
		List<? extends Feature> features = schema.getFeatures();

		Scope scope = new DataFrameScope("X", features, encoder);

		org.dmg.pmml.Expression pmmlExpression = EvaluatableUtil.translateExpression(expr, scope);

		DerivedField derivedField = encoder.createDerivedField(FieldNameUtil.create("expression"), OpType.CONTINUOUS, DataType.DOUBLE, pmmlExpression);

		return RegressionModelUtil.createRegression(Collections.singletonList(new ContinuousFeature(encoder, derivedField)), Collections.singletonList(1d), 0d, RegressionModel.NormalizationMethod.NONE, schema);
	}

	public Expression getExpr(){
		return get("expr", Expression.class);
	}
}