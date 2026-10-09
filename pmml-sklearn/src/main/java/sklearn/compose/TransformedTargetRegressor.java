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

import org.dmg.pmml.Expression;
import org.dmg.pmml.FieldRef;
import org.dmg.pmml.Model;
import org.dmg.pmml.OpType;
import org.dmg.pmml.regression.RegressionModel.NormalizationMethod;
import org.jpmml.converter.FieldNameUtil;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.MultiLabel;
import org.jpmml.converter.PMMLEncoder;
import org.jpmml.converter.ScalarLabel;
import org.jpmml.converter.Schema;
import org.jpmml.converter.Transformation;
import org.jpmml.converter.mining.MiningModelUtil;
import org.jpmml.converter.transformations.AbstractTransformation;
import org.jpmml.python.FunctionUtil;
import org.jpmml.python.Identifiable;
import sklearn.Regressor;
import sklearn.SkLearnRegressor;
import sklearn.preprocessing.FunctionTransformer;

public class TransformedTargetRegressor extends SkLearnRegressor {

	public TransformedTargetRegressor(String module, String name){
		super(module, name);
	}

	@Override
	public int getNumberOfOutputs(){
		Regressor regressor = getRegressor();

		return regressor.getNumberOfOutputs();
	}

	@Override
	public Model encodeModel(Schema schema){
		Regressor regressor = getRegressor();
		FunctionTransformer transformer = getTransformer();

		Identifiable inverseFunc = transformer.getInverseFunc();
		if(inverseFunc == null){
			return regressor.encode(this, schema);
		}

		PMMLEncoder encoder = schema.getEncoder();

		Transformation transformation = new AbstractTransformation(){

			@Override
			public String getName(String name){
				return FieldNameUtil.create("inverseFunc", name);
			}

			@Override
			public Expression createExpression(FieldRef fieldRef){
				return FunctionUtil.encodeFunction(inverseFunc, Collections.singletonList(fieldRef), encoder);
			}
		};

		int numberOfOutputs = getNumberOfOutputs();
		if(numberOfOutputs == 1){
			ScalarLabel scalarLabel = schema.requireScalarLabel();

			Schema segmentSchema = schema.toAnonymousSchema();

			Model model = regressor.encode(this, segmentSchema)
				.setOutput(ModelUtil.createPredictedOutput(FieldNameUtil.create("func", scalarLabel), OpType.CONTINUOUS, regressor.getDataType(), transformation));

			return MiningModelUtil.createRegression(model, NormalizationMethod.NONE, schema);
		} else

		if(numberOfOutputs >= 2){
			MultiLabel multiLabel = schema.requireMultiLabel()
				.expectSize(numberOfOutputs);

			Schema segmentSchema = schema.toLocalRegressorSchema(regressor.getDataType());

			MultiLabel segmentMultiLabel = segmentSchema.requireMultiLabel();

			Model model = regressor.encode(this, segmentSchema)
				.setOutput(ModelUtil.createPredictedOutput("func", OpType.CONTINUOUS, regressor.getDataType(), multiLabel, segmentMultiLabel, transformation));

			return MiningModelUtil.createMultiRegression(model, NormalizationMethod.NONE, schema);
		} else

		{
			throw new IllegalArgumentException();
		}
	}

	public Regressor getRegressor(){
		return getRegressor("regressor_");
	}

	public FunctionTransformer getTransformer(){
		return getTransformer("transformer_", FunctionTransformer.class);
	}
}