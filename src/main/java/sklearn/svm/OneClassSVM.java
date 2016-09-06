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

import org.dmg.pmml.DataType;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.FieldRef;
import org.dmg.pmml.OpType;
import org.dmg.pmml.Output;
import org.dmg.pmml.OutputField;
import org.dmg.pmml.ResultFeature;
import org.dmg.pmml.support_vector_machine.SupportVectorMachineModel;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.converter.Schema;
import org.jpmml.sklearn.FeatureMapper;

public class OneClassSVM extends BaseLibSVMRegressor {

	public OneClassSVM(String module, String name){
		super(module, name);
	}

	@Override
	public Schema createSchema(FeatureMapper featureMapper){

		if(featureMapper.isEmpty()){
			// Ignored
		} else

		{
			featureMapper.initTargetField(createTargetField(), OpType.CONTINUOUS, DataType.DOUBLE, null);
		}

		return super.createSchema(featureMapper);
	}

	@Override
	public SupportVectorMachineModel encodeModel(Schema schema){
		OutputField decisionFunction = ModelUtil.createPredictedField(FieldName.create("decisionFunction"), DataType.DOUBLE);

		OutputField outlier = new OutputField(FieldName.create("outlier"), DataType.BOOLEAN)
			.setOpType(OpType.CATEGORICAL)
			.setResultFeature(ResultFeature.TRANSFORMED_VALUE)
			.setExpression(PMMLUtil.createApply("lessOrEqual", new FieldRef(decisionFunction.getName()), PMMLUtil.createConstant(0d)));

		Output output = new Output()
			.addOutputFields(decisionFunction, outlier);

		SupportVectorMachineModel supportVectorMachineModel = super.encodeModel(schema)
			.setOutput(output);

		return supportVectorMachineModel;
	}

	@Override
	protected FieldName createTargetField(){
		return FieldName.create("distance");
	}
}