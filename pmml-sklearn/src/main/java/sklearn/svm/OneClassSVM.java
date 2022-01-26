/*
 * Copyright (c) 2016 Villu Ruusmann
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
package sklearn.svm;

import java.util.List;

import org.dmg.pmml.DataType;
import org.dmg.pmml.Expression;
import org.dmg.pmml.FieldRef;
import org.dmg.pmml.OpType;
import org.dmg.pmml.Output;
import org.dmg.pmml.OutputField;
import org.dmg.pmml.PMMLFunctions;
import org.dmg.pmml.support_vector_machine.SupportVectorMachineModel;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.converter.Schema;
import org.jpmml.converter.Transformation;
import org.jpmml.converter.transformations.OutlierTransformation;
import sklearn.HasDecisionFunctionField;
import sklearn.SkLearnOutlierTransformation;

public class OneClassSVM extends LibSVMRegressor implements HasDecisionFunctionField {

	public OneClassSVM(String module, String name){
		super(module, name);
	}

	@Override
	public boolean isSupervised(){
		return false;
	}

	@Override
	public SupportVectorMachineModel encodeModel(Schema schema){
		Transformation outlier = new OutlierTransformation(){

			@Override
			public String getName(String name){
				return createFieldName("outlier");
			}

			@Override
			public Expression createExpression(FieldRef fieldRef){
				return PMMLUtil.createApply(PMMLFunctions.LESSOREQUAL, fieldRef, PMMLUtil.createConstant(0d));
			}
		};

		Transformation sklearnOutlier = new SkLearnOutlierTransformation();

		Output output = ModelUtil.createPredictedOutput(getDecisionFunctionField(), OpType.CONTINUOUS, DataType.DOUBLE, outlier, sklearnOutlier);

		List<OutputField> outputFields = output.getOutputFields();

		OutputField decisionFunctionOutputField = outputFields.get(0);

		if(!decisionFunctionOutputField.isFinalResult()){
			decisionFunctionOutputField.setFinalResult(true);
		}

		SupportVectorMachineModel supportVectorMachineModel = super.encodeModel(schema)
			.setOutput(output);

		return supportVectorMachineModel;
	}
}