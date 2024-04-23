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
package sklearn.tree;

import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;

import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.tree.TreeModel;
import org.jpmml.converter.PredicateManager;
import org.jpmml.converter.Schema;
import org.jpmml.converter.ScoreDistributionManager;
import sklearn.Estimator;
import sklearn.HasEstimatorEnsemble;

public class TreeEnsembleUtil {

	private TreeEnsembleUtil(){
	}

	static
	public <E extends Estimator & HasEstimatorEnsemble<T>, T extends Estimator & HasTree> List<TreeModel> encodeTreeModelEnsemble(E estimator, MiningFunction miningFunction, Schema schema){
		List<? extends T> estimators = estimator.getEstimators();

		Schema segmentSchema = schema.toAnonymousSchema();

		PredicateManager predicateManager = new PredicateManager();
		ScoreDistributionManager scoreDistributionManager = (miningFunction == MiningFunction.CLASSIFICATION) ? new ScoreDistributionManager() : null;

		Function<T, TreeModel> function = new Function<T, TreeModel>(){

			@Override
			public TreeModel apply(T estimator){
				TreeModel treeModel = TreeUtil.encodeTreeModel(estimator, miningFunction, predicateManager, scoreDistributionManager, segmentSchema);

				// XXX
				if(estimator.hasFeatureImportances()){
					Schema featureImportanceSchema = TreeUtil.toTreeModelFeatureImportanceSchema(segmentSchema);

					estimator.addFeatureImportances(treeModel, featureImportanceSchema);
				}

				return treeModel;
			}
		};

		return estimators.stream()
			.map(function)
			.collect(Collectors.toList());
	}
}