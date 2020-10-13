/*
 * Copyright (c) 2018 Villu Ruusmann
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
package sklearn.compose;

import java.util.Collections;

import numpy.core.UFunc;
import numpy.core.UFuncUtil;
import org.dmg.pmml.DataType;
import org.dmg.pmml.Expression;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.FieldRef;
import org.dmg.pmml.Model;
import org.dmg.pmml.OpType;
import org.dmg.pmml.regression.RegressionModel.NormalizationMethod;
import org.jpmml.converter.AbstractTransformation;
import org.jpmml.converter.FieldNameUtil;
import org.jpmml.converter.Label;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.Schema;
import org.jpmml.converter.Transformation;
import org.jpmml.converter.mining.MiningModelUtil;
import sklearn.Regressor;
import sklearn.preprocessing.FunctionTransformer;

public class TransformedTargetRegressor extends Regressor {

	public TransformedTargetRegressor(String module, String name){
		super(module, name);
	}

	@Override
	public Model encodeModel(Schema schema){
		Regressor regressor = getRegressor();
		FunctionTransformer transformer = getTransformer();

		UFunc func = transformer.getFunc();
		UFunc inverseFunc = transformer.getInverseFunc();

		if(inverseFunc == null){
			return regressor.encode(schema);
		}

		Label label = schema.getLabel();

		Transformation transformation = new AbstractTransformation(){

			@Override
			public FieldName getName(FieldName name){
				return FieldNameUtil.create("inverseFunc", name);
			}

			@Override
			public Expression createExpression(FieldRef fieldRef){
				return UFuncUtil.encodeUFunc(inverseFunc, Collections.singletonList(fieldRef));
			}
		};

		FieldName name = label.getName();

		Schema segmentSchema = schema.toAnonymousSchema();

		Model model = regressor.encode(segmentSchema)
			.setOutput(ModelUtil.createPredictedOutput(FieldNameUtil.create("func", name), OpType.CONTINUOUS, DataType.DOUBLE, transformation));

		return MiningModelUtil.createRegression(model, NormalizationMethod.NONE, schema);
	}

	public Regressor getRegressor(){
		return get("regressor_", Regressor.class);
	}

	public FunctionTransformer getTransformer(){
		return get("transformer_", FunctionTransformer.class);
	}
}