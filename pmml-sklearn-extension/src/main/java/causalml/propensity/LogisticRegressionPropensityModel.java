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
package causalml.propensity;

import java.util.Arrays;
import java.util.List;

import causalml.CausalMLUtil;
import org.dmg.pmml.Model;
import org.dmg.pmml.OutputField;
import org.dmg.pmml.mining.Segmentation.MissingPredictionTreatment;
import org.jpmml.converter.ContinuousLabel;
import org.jpmml.converter.Feature;
import org.jpmml.converter.Schema;
import org.jpmml.converter.mining.MiningModelUtil;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn.Classifier;
import sklearn.Regressor;
import sklearn.isotonic.IsotonicRegression;

public class LogisticRegressionPropensityModel extends Regressor {

	public LogisticRegressionPropensityModel(String module, String name){
		super(module, name);
	}

	@Override
	public Model encodeModel(Schema schema){
		Boolean calibrate = getCalibrate();
		IsotonicRegression calibrator;
		Object[] clipBounds = getClipBounds();
		Classifier model = getModel();

		if(calibrate){
			calibrator = getCalibrator();
		} else

		{
			List<Number> thresholds = Arrays.asList((Number)clipBounds[0], (Number)clipBounds[1]);

			calibrator = new IsotonicRegression(){

				@Override
				public Boolean getIncreasing(){
					return true;
				}

				@Override
				public String getOutOfBounds(){
					return IsotonicRegression.OUTOFBOUNDS_CLIP;
				}

				@Override
				public List<Number> getXThresholds(){
					return thresholds;
				}

				@Override
				public List<Number> getYThresholds(){
					return thresholds;
				}
			};
		}

		SkLearnEncoder encoder = (SkLearnEncoder)schema.getEncoder();
		ContinuousLabel continuousLabel = schema.requireContinuousLabel();

		Schema classifierSchema = CausalMLUtil.toClassifierSchema(model, schema);

		Model classifierModel = model.encode(classifierSchema);

		OutputField eventOutputField = CausalMLUtil.getProbabilityField(classifierModel);

		List<Feature> eventFeatures = encoder.export(classifierModel, eventOutputField.requireName());

		Schema propensitySchema = new Schema(encoder, continuousLabel, eventFeatures);

		Model propensityModel = calibrator.encode(propensitySchema);

		return MiningModelUtil.createModelChain(Arrays.asList(classifierModel, propensityModel), MissingPredictionTreatment.RETURN_MISSING);
	}

	public Boolean getCalibrate(){
		return getBoolean("calibrate");
	}

	public IsotonicRegression getCalibrator(){
		return getEstimator("calibrator", IsotonicRegression.class);
	}

	public Object[] getClipBounds(){
		return getTuple("clip_bounds");
	}

	public Classifier getModel(){
		return getClassifier("model");
	}
}