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
package sklearn.ensemble.bagging;

import java.util.List;

import org.dmg.pmml.DataType;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.mining.MiningModel;
import org.dmg.pmml.mining.Segmentation;
import org.jpmml.converter.CategoricalLabel;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.Schema;
import sklearn.Classifier;
import sklearn.ensemble.EnsembleClassifier;

public class BaggingClassifier extends EnsembleClassifier {

	public BaggingClassifier(String module, String name){
		super(module, name);
	}

	@Override
	public MiningModel encodeModel(Schema schema){
		List<? extends Classifier> estimators = getEstimators();
		List<List<Integer>> estimatorsFeatures = getEstimatorsFeatures();

		Segmentation.MultipleModelMethod multipleModelMethod = Segmentation.MultipleModelMethod.AVERAGE;

		for(Classifier estimator : estimators){

			if(!estimator.hasProbabilityDistribution()){
				multipleModelMethod = Segmentation.MultipleModelMethod.MAJORITY_VOTE;

				break;
			}
		}

		MiningModel miningModel = BaggingUtil.encodeBagging(estimators, estimatorsFeatures, multipleModelMethod, MiningFunction.CLASSIFICATION, schema)
			.setOutput(ModelUtil.createProbabilityOutput(DataType.DOUBLE, (CategoricalLabel)schema.getLabel()));

		return miningModel;
	}

	public List<List<Integer>> getEstimatorsFeatures(){
		List<?> estimatorsFeatures = (List)get("estimators_features_");

		return BaggingUtil.transformEstimatorsFeatures(estimatorsFeatures);
	}
}