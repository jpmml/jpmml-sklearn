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
package sklearn.ensemble.voting;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.dmg.pmml.DataType;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.Model;
import org.dmg.pmml.mining.MiningModel;
import org.dmg.pmml.mining.Segmentation;
import org.jpmml.converter.CategoricalLabel;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.Schema;
import org.jpmml.converter.mining.MiningModelUtil;
import sklearn.Classifier;
import sklearn.HasEstimatorEnsemble;
import sklearn.SkLearnClassifier;
import sklearn.StepUtil;

public class VotingClassifier extends SkLearnClassifier implements HasEstimatorEnsemble<Classifier> {

	public VotingClassifier(String module, String name){
		super(module, name);
	}

	@Override
	public int getNumberOfFeatures(){
		List<Classifier> estimators = getEstimators();

		return StepUtil.getNumberOfFeatures(estimators);
	}

	@Override
	public MiningModel encodeModel(Schema schema){
		List<Classifier> estimators = getEstimators();
		String voting = getVoting();
		List<Number> weights = getWeights();

		CategoricalLabel categoricalLabel = (CategoricalLabel)schema.getLabel();

		List<Model> models = new ArrayList<>();

		for(Classifier estimator : estimators){
			Model model = estimator.encode(schema);

			models.add(model);
		}

		Segmentation.MultipleModelMethod multipleModelMethod = parseVoting(voting, (weights != null && !weights.isEmpty()));

		MiningModel miningModel = new MiningModel(MiningFunction.CLASSIFICATION, ModelUtil.createMiningSchema(categoricalLabel))
			.setSegmentation(MiningModelUtil.createSegmentation(multipleModelMethod, Segmentation.MissingPredictionTreatment.RETURN_MISSING, models, weights));

		encodePredictProbaOutput(miningModel, DataType.DOUBLE, categoricalLabel);

		return miningModel;
	}

	@Override
	public List<Classifier> getEstimators(){
		return getList("estimators_", Classifier.class);
	}

	public String getVoting(){
		return getEnum("voting", this::getString, Arrays.asList(VotingClassifier.VOTING_HARD, VotingClassifier.VOTING_SOFT));
	}

	public List<Number> getWeights(){
		Object weights = getOptionalObject("weights");

		if((weights == null) || (weights instanceof List)){
			return (List)weights;
		}

		return getNumberArray("weights");
	}

	static
	private Segmentation.MultipleModelMethod parseVoting(String voting, boolean weighted){

		switch(voting){
			case VotingClassifier.VOTING_HARD:
				return (weighted ? Segmentation.MultipleModelMethod.WEIGHTED_MAJORITY_VOTE : Segmentation.MultipleModelMethod.MAJORITY_VOTE);
			case VotingClassifier.VOTING_SOFT:
				return (weighted ? Segmentation.MultipleModelMethod.WEIGHTED_AVERAGE : Segmentation.MultipleModelMethod.AVERAGE);
			default:
				throw new IllegalArgumentException(voting);
		}
	}

	private static final String VOTING_HARD = "hard";
	private static final String VOTING_SOFT = "soft";
}