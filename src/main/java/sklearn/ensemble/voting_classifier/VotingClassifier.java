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
package sklearn.ensemble.voting_classifier;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.dmg.pmml.DefineFunction;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.Model;
import org.dmg.pmml.mining.MiningModel;
import org.dmg.pmml.mining.Segmentation;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.Schema;
import org.jpmml.converter.mining.MiningModelUtil;
import org.jpmml.sklearn.ClassDictUtil;
import sklearn.Classifier;
import sklearn.Estimator;
import sklearn.EstimatorUtil;

public class VotingClassifier extends Classifier {

	public VotingClassifier(String module, String name){
		super(module, name);
	}

	@Override
	public int getNumberOfFeatures(){
		List<? extends Classifier> estimators = getEstimators();

		Estimator estimator = estimators.get(0);

		return estimator.getNumberOfFeatures();
	}

	@Override
	public Set<DefineFunction> encodeDefineFunctions(){
		List<? extends Classifier> estimators = getEstimators();

		Map<String, DefineFunction> uniqueDefineFunctions = new LinkedHashMap<>();

		for(Classifier estimator : estimators){
			Set<DefineFunction> defineFunctions = estimator.encodeDefineFunctions();

			for(DefineFunction defineFunction : defineFunctions){
				uniqueDefineFunctions.put(defineFunction.getName(), defineFunction);
			}
		}

		Set<DefineFunction> result = new LinkedHashSet<>(uniqueDefineFunctions.values());

		return result;
	}

	@Override
	public Model encodeModel(Schema schema){
		List<? extends Classifier> estimators = getEstimators();
		List<? extends Number> weights = getWeights();

		List<Model> models = new ArrayList<>();

		for(Classifier estimator : estimators){
			Model model = estimator.encodeModel(schema);

			models.add(model);
		}

		String voting = getVoting();

		Segmentation.MultipleModelMethod multipleModelMethod = parseVoting(voting, (weights != null && weights.size() > 0));

		MiningModel miningModel = new MiningModel(MiningFunction.CLASSIFICATION, ModelUtil.createMiningSchema(schema))
			.setSegmentation(MiningModelUtil.createSegmentation(multipleModelMethod, models, weights))
			.setOutput(ModelUtil.createProbabilityOutput(schema));

		return miningModel;
	}

	public List<? extends Classifier> getEstimators(){
		List<?> estimators = (List)get("estimators_");

		return EstimatorUtil.asClassifierList(estimators);
	}

	public String getVoting(){
		return (String)get("voting");
	}

	public List<? extends Number> getWeights(){
		Object weights = get("weights");

		if((weights == null) || (weights instanceof List)){
			return (List)weights;
		}

		return (List)ClassDictUtil.getArray(this, "weights");
	}

	static
	private Segmentation.MultipleModelMethod parseVoting(String voting, boolean weighted){

		switch(voting){
			case "hard":
				return (weighted ? Segmentation.MultipleModelMethod.WEIGHTED_MAJORITY_VOTE : Segmentation.MultipleModelMethod.MAJORITY_VOTE);
			case "soft":
				return (weighted ? Segmentation.MultipleModelMethod.WEIGHTED_AVERAGE : Segmentation.MultipleModelMethod.AVERAGE);
			default:
				throw new IllegalArgumentException(voting);
		}
	}
}