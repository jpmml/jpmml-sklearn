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
package sklearn.ensemble.iforest;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import com.google.common.primitives.Ints;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.Model;
import org.dmg.pmml.mining.MiningModel;
import org.dmg.pmml.tree.TreeModel;
import org.jpmml.converter.Label;
import org.jpmml.converter.PredicateManager;
import org.jpmml.converter.Schema;
import org.jpmml.converter.ScoreDistributionManager;
import org.jpmml.python.ClassDictUtil;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn.OutlierDetector;
import sklearn.Regressor;
import sklearn.VersionUtil;
import sklearn.ensemble.EnsembleRegressor;
import sklearn.tree.HasTreeOptions;
import sklearn.tree.Tree;
import sklearn.tree.TreeRegressor;
import sklearn.tree.TreeUtil;

public class IsolationForest extends EnsembleRegressor implements HasIsolationForest, HasTreeOptions, OutlierDetector {

	public IsolationForest(String module, String name){
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
		String sklearnVersion = getSkLearnVersion();
		@SuppressWarnings({"rawtypes", "unchecked"})
		List<TreeRegressor> estimators = (List)getEstimators();
		List<List<Number>> estimatorsFeatures = getEstimatorsFeatures();

		// See https://github.com/scikit-learn/scikit-learn/issues/8549
		boolean corrected = (sklearnVersion != null && VersionUtil.compareVersion(sklearnVersion, "0.19") >= 0);

		// See https://github.com/scikit-learn/scikit-learn/issues/11839
		boolean nodeSampleCorrected = (sklearnVersion != null && VersionUtil.compareVersion(sklearnVersion, "0.21") >= 0);

		PredicateManager predicateManager = new PredicateManager();
		ScoreDistributionManager scoreDistributionManager = new ScoreDistributionManager();

		Schema segmentSchema = schema.toAnonymousSchema();

		List<TreeModel> treeModels = new ArrayList<>();

		for(int i = 0; i < estimators.size(); i++){
			TreeRegressor estimator = estimators.get(i);
			List<Number> estimatorFeatures = estimatorsFeatures.get(i);

			Schema estimatorSchema = segmentSchema.toSubSchema(Ints.toArray(estimatorFeatures));

			Tree tree = estimator.getTree();

			TreeModel treeModel = TreeUtil.encodeTreeModel(estimator, MiningFunction.REGRESSION, predicateManager, scoreDistributionManager, estimatorSchema);

			IsolationForestUtil.transformTreeModel(treeModel, tree, corrected, nodeSampleCorrected);

			ClassDictUtil.clearContent(tree);

			treeModels.add(treeModel);
		}

		return IsolationForestUtil.encodeMiningModel(this, treeModels, corrected, nodeSampleCorrected, schema);
	}

	@Override
	public Schema configureSchema(Schema schema){
		return TreeUtil.configureSchema(this, schema);
	}

	@Override
	public Model configureModel(Model model){
		return TreeUtil.configureModel(this, model);
	}

	@Override
	public Number getDecisionFunctionThreshold(){
		String behaviour = getBehaviour();

		// SkLearn 0.19 or SkLearn 0.24+
		if(behaviour == null){
			return getThreshold();
		} else

		// SkLearn 0.20 through 0.23
		{
			switch(behaviour){
				case IsolationForest.BEHAVIOUR_OLD:
					return getThreshold();
				case IsolationForest.BEHAVIOUR_DEPRECATED:
				case IsolationForest.BEHAVIOUR_NEW:
					return 0d;
				default:
					throw new IllegalArgumentException(behaviour);
			}
		}
	}

	@Override
	public List<Regressor> getEstimators(){
		return getList("estimators_", TreeRegressor.class);
	}

	@Override
	public List<List<Number>> getEstimatorsFeatures(){
		return getArrayList("estimators_features_", Number.class);
	}

	public String getBehaviour(){
		return getOptionalEnum("behaviour", this::getOptionalString, Arrays.asList(IsolationForest.BEHAVIOUR_DEPRECATED, IsolationForest.BEHAVIOUR_NEW, IsolationForest.BEHAVIOUR_OLD));
	}

	@Override
	public Integer getMaxSamples(){
		return getInteger("max_samples_");
	}

	@Override
	public Number getOffset(){

		if(!hasattr("offset_")){
			return 0.5d;
		}

		return getNumber("offset_");
	}

	public Number getThreshold(){

		// SkLearn 0.19
		if(hasattr("threshold_")){
			return getNumber("threshold_");
		} else

		// SkLearn 0.20+
		if(hasattr("_threshold_")){
			return getNumber("_threshold_");
		}

		// SkLearn 0.24+
		return 0d;
	}

	private static final String BEHAVIOUR_DEPRECATED = "deprecated";
	private static final String BEHAVIOUR_NEW = "new";
	private static final String BEHAVIOUR_OLD = "old";
}