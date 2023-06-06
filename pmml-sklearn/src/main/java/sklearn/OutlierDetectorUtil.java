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
package sklearn;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Stream;

import org.dmg.pmml.DataType;
import org.dmg.pmml.Expression;
import org.dmg.pmml.FieldRef;
import org.dmg.pmml.OpType;
import org.dmg.pmml.Output;
import org.dmg.pmml.OutputField;
import org.dmg.pmml.PMMLFunctions;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.converter.Transformation;
import org.jpmml.converter.transformations.OutlierTransformation;

public class OutlierDetectorUtil {

	private OutlierDetectorUtil(){
	}

	static
	public <E extends Estimator & OutlierDetector & HasDecisionFunctionField> Output createPredictedOutput(E estimator){
		Output output = createPredictedOutput(estimator, estimator.getDecisionFunctionField());

		List<OutputField> outputFields = output.getOutputFields();

		OutputField decisionFunctionOutputField = outputFields.get(0);

		if(!decisionFunctionOutputField.isFinalResult()){
			decisionFunctionOutputField.setFinalResult(true);
		}

		return output;
	}

	static
	public <E extends Estimator & OutlierDetector> Output createPredictedOutput(E estimator, String name, Transformation... transformations){
		Transformation outlier = new OutlierTransformation(){

			@Override
			public String getName(String name){
				return estimator.createFieldName("outlier");
			}

			@Override
			public Expression createExpression(FieldRef fieldRef){
				Number threshold = estimator.getDecisionFunctionThreshold();

				return PMMLUtil.createApply(PMMLFunctions.LESSOREQUAL, fieldRef, PMMLUtil.createConstant(threshold));
			}
		};

		Transformation sklearnOutlier = new SkLearnOutlierTransformation();

		transformations = Stream.concat(Arrays.stream(transformations), Stream.of(outlier, sklearnOutlier))
			.toArray(Transformation[]::new);

		return ModelUtil.createPredictedOutput(name, OpType.CONTINUOUS, DataType.DOUBLE, transformations);
	}
}