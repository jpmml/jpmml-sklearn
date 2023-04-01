/*
 * Copyright (c) 2023 Villu Ruusmann
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
package sklearn.calibration;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import com.google.common.collect.Iterables;
import org.dmg.pmml.DataType;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.Model;
import org.dmg.pmml.Output;
import org.dmg.pmml.OutputField;
import org.dmg.pmml.ResultFeature;
import org.dmg.pmml.mining.Segmentation;
import org.dmg.pmml.regression.RegressionModel;
import org.dmg.pmml.regression.RegressionTable;
import org.jpmml.converter.CategoricalLabel;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.Feature;
import org.jpmml.converter.FieldNameUtil;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.Schema;
import org.jpmml.converter.SchemaUtil;
import org.jpmml.converter.mining.MiningModelUtil;
import org.jpmml.converter.regression.RegressionModelUtil;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn.Classifier;
import sklearn.Estimator;
import sklearn.HasEstimator;
import sklearn.Regressor;
import sklearn.isotonic.IsotonicRegression;

public class CalibratedClassifier extends Classifier implements HasEstimator<Classifier> {

	public CalibratedClassifier(String module, String name){
		super(module, name);
	}

	@Override
	public Model encodeModel(Schema schema){
		List<? extends Regressor> calibrators = getCalibrators();
		List<?> classes = getClasses();
		Classifier estimator = getEstimator();
		String method = getMethod();

		switch(method){
			case "isotonic":
			case "sigmoid":
				break;
			default:
				throw new IllegalArgumentException(method);
		}

		SkLearnEncoder encoder = (SkLearnEncoder)schema.getEncoder();

		CategoricalLabel categoricalLabel = (CategoricalLabel)schema.getLabel();
		List<? extends Feature> features = schema.getFeatures();

		Model model = estimator.encodeModel(schema);

		// XXX
		if(model instanceof RegressionModel){
			throw new IllegalArgumentException();
		}

		Output output = model.getOutput();
		if(output == null){
			throw new IllegalArgumentException();
		}

		List<Feature> probabilityFeatures = new ArrayList<>();

		List<OutputField> outputFields = output.getOutputFields();
		for(OutputField outputField : outputFields){
			ResultFeature resultFeature = outputField.getResultFeature();

			switch(resultFeature){
				case PROBABILITY:
					{
						// XXX
						outputField.setName(FieldNameUtil.create(Estimator.FIELD_DECISION_FUNCTION, outputField.getValue()));

						probabilityFeatures.add(new ContinuousFeature(encoder, outputField));
					}
					break;
				default:
					break;
			}
		}

		RegressionModel calibratorModel;

		if(calibrators.size() == 1){
			SchemaUtil.checkSize(2, categoricalLabel);
			SchemaUtil.checkSize(2, probabilityFeatures);

			Regressor calibrator = calibrators.get(0);

			Feature feature = Iterables.getOnlyElement(probabilityFeatures.subList(1, 2));

			Feature calibratedFeature = calibrate(calibrator, model, feature, encoder);

			calibratorModel = RegressionModelUtil.createBinaryLogisticClassification(Collections.singletonList(calibratedFeature), Collections.singletonList(1d), null, RegressionModel.NormalizationMethod.NONE, true, schema);
		} else

		if(calibrators.size() >= 3){
			SchemaUtil.checkSize(calibrators.size(), categoricalLabel);
			SchemaUtil.checkSize(calibrators.size(), probabilityFeatures);

			List<RegressionTable> regressionTables = new ArrayList<>();

			for(int i = 0; i < calibrators.size(); i++){
				Regressor calibrator = calibrators.get(i);
				Feature feature = probabilityFeatures.get(i);

				Feature calibratedFeature = calibrate(calibrator, model, feature, encoder);

				RegressionTable regressionTable = RegressionModelUtil.createRegressionTable(Collections.singletonList(calibratedFeature), Collections.singletonList(1d), null)
					.setTargetCategory(categoricalLabel.getValue(i));

				regressionTables.add(regressionTable);
			}

			calibratorModel = new RegressionModel(MiningFunction.CLASSIFICATION, ModelUtil.createMiningSchema(categoricalLabel), regressionTables)
				.setNormalizationMethod(RegressionModel.NormalizationMethod.SIMPLEMAX)
				.setOutput(ModelUtil.createProbabilityOutput(DataType.DOUBLE, categoricalLabel));
		} else

		{
			throw new IllegalArgumentException();
		}

		return MiningModelUtil.createModelChain(Arrays.asList(model, calibratorModel), Segmentation.MissingPredictionTreatment.RETURN_MISSING);
	}

	@Override
	public List<?> getClasses(){
		return getListLike("classes");
	}

	public List<? extends Regressor> getCalibrators(){
		return getList("calibrators", Regressor.class);
	}

	@Override
	public Classifier getEstimator(){
		return get("estimator", Classifier.class);
	}

	public String getMethod(){
		return getString("method");
	}

	static
	private Feature calibrate(Regressor calibrator, Model model, Feature feature, SkLearnEncoder encoder){
		// XXX
		encoder.export(model, feature.getName());

		Feature calibratedFeature;

		if(calibrator instanceof IsotonicRegression){
			IsotonicRegression isotonicRegression = (IsotonicRegression)calibrator;

			calibratedFeature = Iterables.getOnlyElement(isotonicRegression.encodeFeatures(Collections.singletonList(feature), encoder));
		} else

		if(calibrator instanceof SigmoidCalibration){
			SigmoidCalibration sigmoidCalibration = (SigmoidCalibration)calibrator;

			calibratedFeature = Iterables.getOnlyElement(sigmoidCalibration.encodeFeatures(Collections.singletonList(feature), encoder));
		} else

		{
			throw new IllegalArgumentException();
		}

		DerivedField derivedField = encoder.removeDerivedField(calibratedFeature.getName());

		OutputField outputField = new OutputField(derivedField.requireName(), derivedField.requireOpType(), derivedField.requireDataType())
			.setResultFeature(ResultFeature.TRANSFORMED_VALUE)
			.setExpression(derivedField.requireExpression())
			.setFinalResult(false);

		derivedField = encoder.createDerivedField(model, outputField, true);

		// XXX
		return new ContinuousFeature(encoder, derivedField);
	}
}