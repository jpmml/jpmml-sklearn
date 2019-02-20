/*
 * Copyright (c) 2015 Villu Ruusmann
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
package sklearn.ensemble.gradient_boosting;

import java.util.List;

import numpy.core.ScalarUtil;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.mining.MiningModel;
import org.dmg.pmml.mining.Segmentation;
import org.dmg.pmml.tree.TreeModel;
import org.jpmml.converter.ContinuousLabel;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.Schema;
import org.jpmml.converter.mining.MiningModelUtil;
import sklearn.Estimator;
import sklearn.HasEstimatorEnsemble;
import sklearn.tree.HasTreeOptions;
import sklearn.tree.TreeModelUtil;
import sklearn.tree.TreeRegressor;

public class GradientBoostingUtil {

	private GradientBoostingUtil(){
	}

	static
	public <E extends Estimator & HasEstimatorEnsemble<TreeRegressor>> Number getLearningRate(E estimator){
		Object learningRate = ScalarUtil.decode(estimator.get("learning_rate"));

		return (Number)learningRate;
	}

	static
	public <E extends Estimator & HasEstimatorEnsemble<TreeRegressor> & HasTreeOptions> MiningModel encodeGradientBoosting(E estimator, Number initialPrediction, Number learningRate, Schema schema){
		ContinuousLabel continuousLabel = (ContinuousLabel)schema.getLabel();

		List<TreeModel> treeModels = TreeModelUtil.encodeTreeModelSegmentation(estimator, MiningFunction.REGRESSION, schema);

		MiningModel miningModel = new MiningModel(MiningFunction.REGRESSION, ModelUtil.createMiningSchema(continuousLabel))
			.setSegmentation(MiningModelUtil.createSegmentation(Segmentation.MultipleModelMethod.SUM, treeModels))
			.setTargets(ModelUtil.createRescaleTargets(learningRate, initialPrediction, continuousLabel));

		return TreeModelUtil.transform(estimator, miningModel);
	}
}