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

import java.util.Collections;
import java.util.List;

import org.dmg.pmml.DataType;
import org.dmg.pmml.Decision;
import org.dmg.pmml.Decisions;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.Expression;
import org.dmg.pmml.Model;
import org.dmg.pmml.OpType;
import org.dmg.pmml.OutputField;
import org.dmg.pmml.ResultFeature;
import org.jpmml.converter.Feature;
import org.jpmml.converter.FeatureUtil;
import org.jpmml.converter.SchemaUtil;
import org.jpmml.python.TupleUtil;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn.Transformer;
import sklearn2pmml.preprocessing.ExpressionTransformer;

public class BusinessDecisionTransformer extends Transformer {

	public BusinessDecisionTransformer(String module, String name){
		super(module, name);
	}

	@Override
	public List<Feature> encodeFeatures(List<Feature> features, SkLearnEncoder encoder){
		Transformer transformer = getTransformer();
		String businessProblem = getBusinessProblem();
		List<Object[]> decisions = getDecisions();

		if(!encoder.hasModel()){
			throw new IllegalStateException("Model is not defined");
		}

		Model model = encoder.getModel();

		if(transformer != null){
			features = transformer.encode(features, encoder);
		}

		SchemaUtil.checkSize(1, features);

		Feature feature = features.get(0);

		DerivedField derivedField = (DerivedField)feature.getField();

		DataType dataType = derivedField.requireDataType();
		OpType opType = derivedField.requireOpType();

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

		Expression expression;

		if(isEmbedded()){
			expression = derivedField.getExpression();

			encoder.removeDerivedField(derivedField.requireName());
		} else

		{
			expression = feature.ref();
		}

		Decisions pmmlDecisions = new Decisions()
			.setBusinessProblem(businessProblem);

		for(Object[] decision : decisions){
			Decision pmmlDecision = new Decision()
				.setValue(TupleUtil.extractElement(decision, 0, Object.class))
				.setDescription(TupleUtil.extractElement(decision, 1, String.class));

			pmmlDecisions.addDecisions(pmmlDecision);
		}

		OutputField outputField = new OutputField(createFieldName("decision", feature), opType, dataType)
			.setResultFeature(ResultFeature.DECISION)
			.setFinalResult(true)
			.setExpression(expression)
			.setDecisions(pmmlDecisions);

		DerivedField decisionDerivedField = encoder.createDerivedField(model, outputField, true);

		feature = FeatureUtil.createFeature(decisionDerivedField, encoder);

		return Collections.singletonList(feature);
	}

	public String getBusinessProblem(){
		return getString("business_problem");
	}

	public List<Object[]> getDecisions(){
		return getTupleList("decisions");
	}

	public Transformer getTransformer(){

		// SkLearn2PMML 0.80.0
		if(hasattr("expr")){
			String expr = getString("expr");
			Object dtype = getOptionalObject("dtype");

			ExpressionTransformer expressionTransformer = new ExpressionTransformer()
				.setExpr(expr)
				.setDType(dtype);

			return expressionTransformer;
		}

		// SkLearn2PMML 0.81.0+
		return getOptional("transformer_", Transformer.class);
	}

	private boolean isEmbedded(){

		// SkLearn2PMML 0.80.0
		if(hasattr("expr")){
			return true;
		}

		// SkLearn2PMML 0.81.0+
		Object transformer = getOptionalObject("transformer");
		if(transformer != null){

			if(transformer instanceof String){
				return true;
			} else

			// Accept simple transformers, reject complex transformers and pipelines
			if(transformer instanceof Transformer){
				return true;
			}
		}

		return false;
	}
}