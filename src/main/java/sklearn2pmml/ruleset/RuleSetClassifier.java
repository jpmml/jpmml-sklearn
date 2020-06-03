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
package sklearn2pmml.ruleset;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

import org.dmg.pmml.FieldName;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.Predicate;
import org.dmg.pmml.rule_set.RuleSelectionMethod;
import org.dmg.pmml.rule_set.RuleSet;
import org.dmg.pmml.rule_set.RuleSetModel;
import org.dmg.pmml.rule_set.SimpleRule;
import org.jpmml.converter.Feature;
import org.jpmml.converter.Label;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.Schema;
import org.jpmml.python.DataFrameScope;
import org.jpmml.python.PredicateTranslator;
import org.jpmml.python.Scope;
import org.jpmml.python.TupleUtil;
import sklearn.Classifier;

public class RuleSetClassifier extends Classifier {

	public RuleSetClassifier(String module, String name){
		super(module, name);
	}

	@Override
	public boolean hasProbabilityDistribution(){
		return false;
	}

	@Override
	public List<String> getClasses(){
		String defaultScore = getDefaultScore();
		List<Object[]> rules = getRules();

		Set<String> scores = new LinkedHashSet<>();

		if(defaultScore != null){
			scores.add(defaultScore);
		}

		for(Object[] rule : rules){
			String score = TupleUtil.extractElement(rule, 1, String.class);

			scores.add(score);
		}

		List<String> result = new ArrayList<>(scores);

		Collections.sort(result, String.CASE_INSENSITIVE_ORDER);

		return result;
	}

	@Override
	public RuleSetModel encodeModel(Schema schema){
		String defaultScore = getDefaultScore();
		List<Object[]> rules = getRules();

		Label label = schema.getLabel();
		List<? extends Feature> features = schema.getFeatures();

		RuleSelectionMethod ruleSelectionMethod = new RuleSelectionMethod(RuleSelectionMethod.Criterion.FIRST_HIT);

		RuleSet ruleSet = new RuleSet()
			.addRuleSelectionMethods(ruleSelectionMethod);

		if(defaultScore != null){
			ruleSet
				.setDefaultConfidence(1d)
				.setDefaultScore(defaultScore);
		}

		Scope scope = new DataFrameScope(FieldName.create("X"), features);

		for(Object[] rule : rules){
			String predicate = TupleUtil.extractElement(rule, 0, String.class);
			String score = TupleUtil.extractElement(rule, 1, String.class);

			Predicate pmmlPredicate = PredicateTranslator.translate(predicate, scope);

			SimpleRule simpleRule = new SimpleRule(score, pmmlPredicate);

			ruleSet.addRules(simpleRule);
		}

		RuleSetModel ruleSetModel = new RuleSetModel(MiningFunction.CLASSIFICATION, ModelUtil.createMiningSchema(label), ruleSet);

		return ruleSetModel;
	}

	public String getDefaultScore(){
		return getOptionalString("default_score");
	}

	public List<Object[]> getRules(){
		return getTupleList("rules");
	}
}