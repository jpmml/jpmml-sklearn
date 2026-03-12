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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

import causalml.CausalMLUtil;
import causalml.propensity.PropensityModel;
import org.dmg.pmml.DataType;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.Model;
import org.dmg.pmml.mining.MiningModel;
import org.dmg.pmml.mining.Segmentation;
import org.dmg.pmml.regression.PredictorTerm;
import org.dmg.pmml.regression.RegressionModel;
import org.dmg.pmml.regression.RegressionTable;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.ContinuousLabel;
import org.jpmml.converter.FieldNameUtil;
import org.jpmml.converter.Label;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.ScalarLabelUtil;
import org.jpmml.converter.Schema;
import org.jpmml.converter.mining.MiningModelUtil;
import org.jpmml.python.ClassDictUtil;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn.Regressor;
import sklearn.SkLearnMethods;

abstract
public class BaseXLearner extends BaseLearner<Regressor> {

	public BaseXLearner(String module, String name){
		super(module, name);
	}

	@Override
	public Class<Regressor> getEstimatorClass(){
		return Regressor.class;
	}

	@Override
	public Model encodeModel(Schema schema){
		List<String> treatmentGroups = getTreatmentGroups();
		Map<String, Regressor> modelsTauC = getModelsTauC();
		Map<String, Regressor> modelsTauT = getModelsTauT();
		Map<String, PropensityModel> propensityModels = getPropensityModels();

		Label label = schema.getLabel();

		List<ContinuousLabel> continuousLabels = ScalarLabelUtil.toScalarLabels(ContinuousLabel.class, label);

		ClassDictUtil.checkSize(continuousLabels, treatmentGroups, modelsTauT.entrySet(), modelsTauC.entrySet(), propensityModels.entrySet());

		List<Model> binaryModels = new ArrayList<>();

		for(int i = 0; i < continuousLabels.size(); i++){
			ContinuousLabel continuousLabel = continuousLabels.get(i);
			String treatmentGroup = treatmentGroups.get(i);

			PropensityModel propensityEstimator = propensityModels.get(treatmentGroup);

			Schema propensitySchema = CausalMLUtil.toRegressorSchema(propensityEstimator, schema);

			Model propensityModel = propensityEstimator.encode(treatmentGroup, propensitySchema);

			Regressor controlEffectRegressor = modelsTauC.get(treatmentGroup);
			Regressor treatmentEffectRegressor = modelsTauT.get(treatmentGroup);

			Schema segmentSchema = schema.toRelabeledSchema(continuousLabel);

			Model controlEffectModel = encodeEstimator(Role.CONTROL, controlEffectRegressor, segmentSchema);
			Model treatmentEffectModel = encodeEstimator(Role.TREATMENT, treatmentEffectRegressor, segmentSchema);

			Model binaryModel = encodeBinaryModel(treatmentGroup, propensityModel, controlEffectModel, treatmentEffectModel, segmentSchema);

			binaryModels.add(binaryModel);
		}

		return BaseLearnerUtil.encodeModel(binaryModels);
	}

	protected MiningModel encodeBinaryModel(String treatmentGroup, Model propensityModel, Model controlModel, Model treatmentModel, Schema schema){
		SkLearnEncoder encoder = (SkLearnEncoder)schema.getEncoder();

		ContinuousLabel continuousLabel = new ContinuousLabel(null, DataType.DOUBLE);

		ContinuousFeature propensityFeature = (ContinuousFeature)encoder.exportPrediction(propensityModel, FieldNameUtil.create(SkLearnMethods.PREDICT, treatmentGroup, "propensity"), continuousLabel);

		ContinuousFeature controlFeature = (ContinuousFeature)encoder.exportPrediction(controlModel, FieldNameUtil.create(SkLearnMethods.PREDICT, treatmentGroup, "control"), continuousLabel);
		ContinuousFeature treatmentFeature = (ContinuousFeature)encoder.exportPrediction(treatmentModel, FieldNameUtil.create(SkLearnMethods.PREDICT, treatmentGroup, "treatment"), continuousLabel);

		// "(1 - p) * treatment + p * control"
		RegressionTable regressionTable = new RegressionTable(0d)
			.addPredictorTerms(new PredictorTerm(1d, null)
				.addFieldRefs(treatmentFeature.ref())
			)
			.addPredictorTerms(new PredictorTerm(-1d, null)
				.addFieldRefs(propensityFeature.ref(), treatmentFeature.ref())
			)
			.addPredictorTerms(new PredictorTerm(1d, null)
				.addFieldRefs(propensityFeature.ref(), controlFeature.ref())
			);

		RegressionModel regressionModel = new RegressionModel(MiningFunction.REGRESSION, ModelUtil.createMiningSchema(schema), null)
			.addRegressionTables(regressionTable);

		return MiningModelUtil.createModelChain(Arrays.asList(propensityModel, treatmentModel, controlModel, regressionModel), Segmentation.MissingPredictionTreatment.RETURN_MISSING);
	}

	public Map<String, Regressor> getModelsTauC(){
		return getModels("models_tau_c");
	}

	public Map<String, Regressor> getModelsTauT(){
		return getModels("models_tau_t");
	}

	public Map<String, PropensityModel> getPropensityModels(){
		return getModels("propensity_model", PropensityModel.class);
	}
}