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
package sktree.ensemble.forest;

import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;

import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.mining.MiningModel;
import org.dmg.pmml.mining.Segmentation;
import org.dmg.pmml.tree.TreeModel;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.Schema;
import org.jpmml.converter.mining.MiningModelUtil;
import sklearn.Estimator;
import sklearn.HasEstimatorEnsemble;
import sklearn.tree.HasTree;

public class ObliqueForestUtil {

	private ObliqueForestUtil(){
	}

	static
	public <E extends Estimator & HasEstimatorEnsemble<T>, T extends Estimator & HasTree> MiningModel encodeBaseObliqueForest(E estimator, MiningFunction miningFunction, Segmentation.MultipleModelMethod multipleModelMethod, Schema schema){
		List<? extends T> estimators = estimator.getEstimators();

		Schema segmentSchema = schema.toAnonymousSchema();

		Function<T, TreeModel> function = new Function<T, TreeModel>(){

			@Override
			public TreeModel apply(T estimator){
				int i = estimators.indexOf(estimator);

				if(i < 0){
					throw new IllegalArgumentException();
				}

				return (TreeModel)estimator.encode((i + 1), segmentSchema);
			}
		};

		List<TreeModel> treeModels = estimators.stream()
			.map(function)
			.collect(Collectors.toList());

		MiningModel miningModel = new MiningModel(miningFunction, ModelUtil.createMiningSchema(schema.getLabel()))
			.setSegmentation(MiningModelUtil.createSegmentation(multipleModelMethod, Segmentation.MissingPredictionTreatment.RETURN_MISSING, treeModels));

		return miningModel;

	}
}