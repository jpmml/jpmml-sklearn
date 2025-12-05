/*
 * Copyright (c) 2024 Villu Ruusmann
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
package sktree.ensemble.eiforest;

import java.util.ArrayList;
import java.util.List;

import com.google.common.primitives.Ints;
import org.dmg.pmml.mining.MiningModel;
import org.dmg.pmml.tree.TreeModel;
import org.jpmml.converter.Label;
import org.jpmml.converter.PredicateManager;
import org.jpmml.converter.Schema;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn.OutlierDetector;
import sklearn.Regressor;
import sklearn.ensemble.iforest.HasIsolationForest;
import sklearn.ensemble.iforest.IsolationForestUtil;
import sklearn.tree.Tree;
import sktree.tree.ObliqueDecisionTreeRegressor;
import sktree.tree.ProjectionManager;

public class ExtendedIsolationForest extends Regressor implements HasIsolationForest, OutlierDetector {

	public ExtendedIsolationForest(String module, String name){
		super(module, name);
	}

	@Override
	public int getNumberOfOutputs(){
		return 0;
	}

	@Override
	public boolean isSupervised(){
		return false;
	}

	@Override
	public Label encodeLabel(List<String> names, SkLearnEncoder encoder){
		throw new UnsupportedOperationException();
	}

	@Override
	public MiningModel encodeModel(Schema schema){
		@SuppressWarnings({"rawtypes", "unchecked"})
		List<ObliqueDecisionTreeRegressor> estimators = (List)getEstimators();
		List<List<Number>> estimatorsFeatures = getEstimatorsFeatures();

		// Assume SkLearn 1.3.0+
		boolean corrected = true;
		boolean nodeSampleCorrected = true;

		Schema segmentSchema = schema.toAnonymousSchema();

		PredicateManager predicateManager = new PredicateManager();
		ProjectionManager projectionManager = new ProjectionManager();

		List<TreeModel> treeModels = new ArrayList<>();

		for(int i = 0; i < estimators.size(); i++){
			ObliqueDecisionTreeRegressor estimator = estimators.get(i);
			List<Number> estimatorFeatures = estimatorsFeatures.get(i);

			Schema estimatorSchema = segmentSchema.toSubSchema(Ints.toArray(estimatorFeatures));

			Tree tree = estimator.getTree();

			TreeModel treeModel;

			try {
				estimator.setPMMLSegmentId((i + 1));

				treeModel = estimator.encodeModel(predicateManager, projectionManager, estimatorSchema);
			} finally {
				estimator.setPMMLSegmentId(null);
			}

			IsolationForestUtil.transformTreeModel(treeModel, tree, corrected, nodeSampleCorrected);

			treeModels.add(treeModel);
		}

		return IsolationForestUtil.encodeMiningModel(this, treeModels, corrected, nodeSampleCorrected, schema);
	}

	@Override
	public List<Regressor> getEstimators(){
		return getEstimatorList("estimators_", ObliqueDecisionTreeRegressor.class);
	}

	@Override
	public List<List<Number>> getEstimatorsFeatures(){
		return getArrayList("estimators_features_", Number.class);
	}

	@Override
	public Integer getMaxSamples(){
		return getInteger("_max_samples");
	}

	@Override
	public Number getOffset(){
		return getNumber("offset_");
	}
}