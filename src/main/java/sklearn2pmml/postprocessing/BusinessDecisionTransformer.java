/*
 * Copyright (c) 2021 Villu Ruusmann
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
package sklearn2pmml.postprocessing;

import java.util.List;

import org.dmg.pmml.DataType;
import org.dmg.pmml.Decision;
import org.dmg.pmml.Decisions;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.Expression;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.Model;
import org.dmg.pmml.OpType;
import org.dmg.pmml.OutputField;
import org.dmg.pmml.ResultFeature;
import org.jpmml.python.TupleUtil;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn2pmml.preprocessing.ExpressionTransformer;

public class BusinessDecisionTransformer extends ExpressionTransformer {

	public BusinessDecisionTransformer(String module, String name){
		super(module, name);
	}

	@Override
	protected DerivedField encodeDerivedField(FieldName name, OpType opType, DataType dataType, Expression expression, SkLearnEncoder encoder){
		String businessProblem = getBusinessProblem();
		List<Object[]> decisions = getDecisions();

		Model model = encoder.getModel();
		if(model == null){
			throw new IllegalArgumentException("Model is undefined");
		}

		switch(opType){
			case CONTINUOUS:
				opType = OpType.CATEGORICAL;
				break;
			case CATEGORICAL:
			case ORDINAL:
				break;
			default:
				break;
		}

		Decisions pmmlDecisions = new Decisions()
			.setBusinessProblem(businessProblem);

		for(Object[] decision : decisions){
			Decision pmmlDecision = new Decision()
				.setValue(TupleUtil.extractElement(decision, 0, String.class))
				.setDescription(TupleUtil.extractElement(decision, 1, String.class));

			pmmlDecisions.addDecisions(pmmlDecision);
		}

		OutputField outputField = new OutputField(name, opType, dataType)
			.setResultFeature(ResultFeature.DECISION)
			.setFinalResult(true)
			.setExpression(expression)
			.setDecisions(pmmlDecisions);

		return encoder.createDerivedField(model, outputField, true);
	}

	public String getBusinessProblem(){
		return getString("business_problem");
	}

	public List<Object[]> getDecisions(){
		return getTupleList("decisions");
	}

	@Override
	public String getExpr(){
		return getString("expr");
	}
}