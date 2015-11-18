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

import org.dmg.pmml.MiningFunctionType;
import org.dmg.pmml.MiningModel;
import org.dmg.pmml.Output;
import org.dmg.pmml.PMML;
import org.dmg.pmml.TransformationDictionary;
import org.jpmml.sklearn.Schema;
import sklearn.Classifier;
import sklearn.EstimatorUtil;

public class BaggingClassifier extends Classifier {

	public BaggingClassifier(String module, String name){
		super(module, name);
	}

	@Override
	public MiningModel encodeModel(Schema schema){
		List<Classifier> estimators = getEstimators();
		List<List<Integer>> estimatorsFeatures = getEstimatorsFeatures();

		Output output = EstimatorUtil.encodeClassifierOutput(schema);

		MiningModel miningModel = BaggingUtil.encodeBagging(estimators, estimatorsFeatures, MiningFunctionType.CLASSIFICATION, schema)
			.setOutput(output);

		return miningModel;
	}

	@Override
	public PMML encodePMML(Schema schema){
		PMML pmml = super.encodePMML(schema);

		TransformationDictionary transformationDictionary = new TransformationDictionary()
			.addDefineFunctions(EstimatorUtil.encodeAdaBoostFunction(), EstimatorUtil.encodeLogitFunction());

		pmml.setTransformationDictionary(transformationDictionary);

		return pmml;
	}

	public List<Classifier> getEstimators(){
		List<?> estimators = (List)get("estimators_");

		return BaggingUtil.transformEstimators(estimators, Classifier.class);
	}

	public List<List<Integer>> getEstimatorsFeatures(){
		List<?> estimatorsFeatures = (List)get("estimators_features_");

		return BaggingUtil.transformEstimatorsFeatures(estimatorsFeatures);
	}
}