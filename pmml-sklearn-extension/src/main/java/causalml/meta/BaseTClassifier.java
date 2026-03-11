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
import org.dmg.pmml.OutputField;
import org.dmg.pmml.mining.MiningModel;
import org.dmg.pmml.mining.Segmentation;
import org.dmg.pmml.regression.RegressionModel;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.ModelEncoder;
import org.jpmml.converter.Schema;
import org.jpmml.converter.mining.MiningModelUtil;
import org.jpmml.converter.regression.RegressionModelUtil;
import sklearn.Classifier;

public class BaseTClassifier extends BaseTLearner<Classifier> {

	public BaseTClassifier(String module, String name){
		super(module, name);
	}

	@Override
	public Class<Classifier> getEstimatorClass(){
		return Classifier.class;
	}

	@Override
	public Model encodeEstimator(Role role, Classifier classifier, Schema schema){
		Schema classifierSchema = CausalMLUtil.toClassifierSchema(classifier, schema);

		return classifier.encode(role.getSegmentId(), classifierSchema);
	}

	@Override
	protected MiningModel encodeBinaryModel(Model controlModel, Model treatmentModel, Schema schema){
		ModelEncoder encoder = schema.getEncoder();

		OutputField treatmentOutputField = CausalMLUtil.getProbabilityField(treatmentModel);
		OutputField controlOutputField = CausalMLUtil.getProbabilityField(controlModel);

		RegressionModel regressionModel = RegressionModelUtil.createRegression(Arrays.asList(new ContinuousFeature(encoder, treatmentOutputField), new ContinuousFeature(encoder, controlOutputField)), Arrays.asList(1, -1), null, RegressionModel.NormalizationMethod.NONE, schema);

		return MiningModelUtil.createModelChain(Arrays.asList(treatmentModel, controlModel, regressionModel), Segmentation.MissingPredictionTreatment.RETURN_MISSING);
	}
}