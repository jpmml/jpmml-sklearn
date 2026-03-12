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
package causalml.meta;

import java.util.Arrays;

import causalml.CausalMLUtil;
import org.dmg.pmml.Model;
import org.dmg.pmml.mining.MiningModel;
import org.jpmml.converter.Schema;
import sklearn.Classifier;
import sklearn.EstimatorCastException;
import sklearn.EstimatorUtil;
import sklearn.ensemble.forest.ForestClassifier;
import sklearn.tree.TreeClassifier;

public class BaseSClassifier extends BaseSLearner<Classifier> {

	public BaseSClassifier(String module, String name){
		super(module, name);
	}

	@Override
	public Class<Classifier> getEstimatorClass(){
		return Classifier.class;
	}

	@Override
	public Model encodeEstimator(Role role, Classifier classifier, Schema schema){

		if(classifier instanceof TreeClassifier){
			TreeClassifier treeClassifier = (TreeClassifier)classifier;
		} else

		if(classifier instanceof ForestClassifier){
			ForestClassifier forestClassifier = (ForestClassifier)classifier;
		} else

		{
			throw new EstimatorCastException(classifier, Arrays.asList(TreeClassifier.class, ForestClassifier.class));
		}

		Schema classifierSchema = CausalMLUtil.toClassifierSchema(classifier, schema);

		return EstimatorUtil.encodeNativeLike(classifier, role.getSegmentId(), classifierSchema);
	}

	@Override
	protected MiningModel encodeBinaryModel(Model controlModel, Model treatmentModel, Schema schema){
		return BaseLearnerUtil.encodeBinaryClassifier(controlModel, treatmentModel, schema);
	}
}