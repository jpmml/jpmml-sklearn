/*
 * Copyright (c) 2026 Villu Ruusmann
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
package sklearn.ensemble.weight_boosting;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

import org.dmg.pmml.DataType;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.OpType;
import org.dmg.pmml.ScoreDistribution;
import org.dmg.pmml.Visitor;
import org.dmg.pmml.VisitorAction;
import org.dmg.pmml.mining.MiningModel;
import org.dmg.pmml.mining.Segmentation;
import org.dmg.pmml.mining.Segmentation.MultipleModelMethod;
import org.dmg.pmml.regression.RegressionModel;
import org.dmg.pmml.tree.Node;
import org.dmg.pmml.tree.TreeModel;
import org.jpmml.converter.CategoricalLabel;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.Schema;
import org.jpmml.converter.mining.MiningModelUtil;
import org.jpmml.model.visitors.AbstractVisitor;
import sklearn.Classifier;
import sklearn.ensemble.EnsembleClassifier;
import sklearn.tree.TreeClassifier;

public class AdaBoostClassifier extends EnsembleClassifier {

	public AdaBoostClassifier(String module, String name){
		super(module, name);
	}

	@Override
	public MiningModel encodeModel(Schema schema){
		@SuppressWarnings("unused")
		String algorithm = getAlgorithm();
		List<Classifier> estimators = getEstimators();
		List<Number> estimatorWeights = getEstimatorWeights();

		CategoricalLabel categoricalLabel = schema.requireCategoricalLabel();

		Schema segmentSchema = schema.toAnonymousSchema();

		List<TreeModel> treeModels = new ArrayList<>();

		for(Classifier estimator : estimators){
			TreeModel treeModel = (TreeModel)estimator.encode(segmentSchema);

			treeModel
				.setMiningFunction(MiningFunction.REGRESSION)
				.setAlgorithmName(null)
				.setOutput(null);

			Visitor visitor = new AbstractVisitor(){

				@Override
				public VisitorAction visit(Node node){

					if(node.hasScoreDistributions()){
						Object score = node.requireScore();
						List<ScoreDistribution> scoreDistributions = node.getScoreDistributions();

						if(scoreDistributions.size() != 2){
							throw new IllegalArgumentException();
						} // End if

						if(Objects.equals(score, categoricalLabel.getValue(0))){
							node
								.setScore(-2)
								.setRecordCount(null);
						} else

						if(Objects.equals(score, categoricalLabel.getValue(1))){
							node
								.setScore(2)
								.setRecordCount(null);
						} else

						{
							throw new IllegalArgumentException();
						}

						scoreDistributions.clear();
					}

					return super.visit(node);
				}
			};
			visitor.applyTo(treeModel);

			treeModels.add(treeModel);
		}

		MiningModel miningModel = new MiningModel(MiningFunction.REGRESSION, ModelUtil.createMiningSchema(segmentSchema))
			.setSegmentation(MiningModelUtil.createSegmentation(MultipleModelMethod.WEIGHTED_AVERAGE, Segmentation.MissingPredictionTreatment.RETURN_MISSING, treeModels, estimatorWeights))
			.setOutput(ModelUtil.createPredictedOutput("adaValue", OpType.CONTINUOUS, DataType.DOUBLE));

		return MiningModelUtil.createBinaryLogisticClassification(miningModel, 1d, 0d, RegressionModel.NormalizationMethod.LOGIT, true, schema);
	}

	public String getAlgorithm(){
		return getOptionalEnum("algorithm", this::getOptionalString, Arrays.asList(AdaBoostClassifier.ALGORITHM_SAMME));
	}

	@Override
	public List<Classifier> getEstimators(){
		return getEstimatorList("estimators_", TreeClassifier.class);
	}

	public List<Number> getEstimatorWeights(){
		return getNumberArray("estimator_weights_");
	}

	private static final String ALGORITHM_SAMME = "SAMME";
}