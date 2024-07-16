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
import org.dmg.pmml.OpType;
import org.dmg.pmml.Output;
import org.dmg.pmml.OutputField;
import org.dmg.pmml.ResultFeature;
import org.dmg.pmml.mining.MiningModel;
import org.dmg.pmml.mining.Segment;
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
import sklearn.Calibrator;
import sklearn.Classifier;
import sklearn.Estimator;
import sklearn.EstimatorUtil;
import sklearn.HasEstimator;
import sklearn.SkLearnClassifier;
import sklearn.ensemble.gradient_boosting.GradientBoostingClassifier;
import sklearn.linear_model.LinearClassifier;

public class CalibratedClassifier extends SkLearnClassifier implements HasEstimator<Classifier> {

	public CalibratedClassifier(String module, String name){
		super(module, name);
	}

	@Override
	public Model encodeModel(Schema schema){
		List<? extends Calibrator> calibrators = getCalibrators();
		List<?> classes = getClasses();
		Classifier estimator = getEstimator();
		@SuppressWarnings("unused")
		String method = getMethod();

		SkLearnEncoder encoder = (SkLearnEncoder)schema.getEncoder();
		CategoricalLabel categoricalLabel = (CategoricalLabel)schema.getLabel();
		List<? extends Feature> features = schema.getFeatures();

		Model model = estimator.encode(schema);

		List<Model> models = new ArrayList<>();

		List<Feature> decisionFunctionFeatures = new ArrayList<>();

		if((estimator instanceof GradientBoostingClassifier) || (estimator instanceof LinearClassifier)){

			if(model instanceof MiningModel){
				MiningModel miningModel = (MiningModel)model;

				Segmentation segmentation = miningModel.requireSegmentation();

				List<Segment> segments = segmentation.requireSegments();

				List<Segment> regressorSegments = segments.subList(0, segments.size() - 1);
				for(Segment regressorSegment : regressorSegments){
					Model decisionFunctionModel = regressorSegment.requireModel();

					if(decisionFunctionModel.requireMiningFunction() != MiningFunction.REGRESSION){
						throw new IllegalArgumentException();
					}

					Output output = decisionFunctionModel.getOutput();
					if(output == null || !output.hasOutputFields()){
						throw new IllegalArgumentException();
					}

					List<OutputField> outputFields = output.getOutputFields();

					OutputField outputField;

					if(estimator instanceof LinearClassifier){
						RegressionModel regressionModel = (RegressionModel)decisionFunctionModel;

						regressionModel.setNormalizationMethod(RegressionModel.NormalizationMethod.NONE);

						outputField = Iterables.getOnlyElement(outputFields);
					} else

					if(estimator instanceof GradientBoostingClassifier){

						if(outputFields.size() == 1){
							// Ignored
						} else

						if(outputFields.size() == 2){
							outputFields.remove(1);
						} else

						{
							throw new IllegalArgumentException();
						}

						outputField = Iterables.getOnlyElement(outputFields);
					} else

					{
						throw new IllegalArgumentException();
					} // End if

					if(outputField.getResultFeature() != ResultFeature.PREDICTED_VALUE){
						throw new IllegalArgumentException();
					}

					// XXX
					outputField.setName(getDecisionFunctionField(outputField.requireName()));

					models.add(decisionFunctionModel);

					decisionFunctionFeatures.add(new ContinuousFeature(encoder, outputField));
				}

				List<Segment> classifierSegments = Collections.singletonList(segments.get(segments.size() - 1));
				for(Segment classifierSegment : classifierSegments){
					RegressionModel normalizerModel = (RegressionModel)classifierSegment.requireModel();

					if(normalizerModel.requireMiningFunction() != MiningFunction.CLASSIFICATION){
						throw new IllegalArgumentException();
					}
				}
			} else

			if(model instanceof RegressionModel){
				RegressionModel regressionModel = (RegressionModel)model;

				List<RegressionTable> regressionTables = regressionModel.getRegressionTables();

				// XXX
				if(categoricalLabel.size() == 2){
					regressionTables = regressionTables.subList(0, 1);
				}

				for(RegressionTable regressionTable : regressionTables){
					OutputField outputField = ModelUtil.createPredictedField(getDecisionFunctionField(regressionTable.requireTargetCategory()), OpType.CONTINUOUS, DataType.DOUBLE)
						.setFinalResult(false);

					Output output = new Output()
						.addOutputFields(outputField);

					regressionTable.setTargetCategory(null);

					Model decisionFunctionModel = new RegressionModel(MiningFunction.REGRESSION, ModelUtil.createMiningSchema(null), null)
						.setNormalizationMethod(RegressionModel.NormalizationMethod.NONE)
						.addRegressionTables(regressionTable)
						.setOutput(output);

					models.add(decisionFunctionModel);

					decisionFunctionFeatures.add(new ContinuousFeature(encoder, outputField));
				}
			} else

			{
				throw new IllegalArgumentException();
			}
		} else

		{
			Output output = EstimatorUtil.getFinalOutput(model);
			if(output == null){
				throw new IllegalArgumentException();
			}

			List<OutputField> outputFields = output.getOutputFields();
			for(OutputField outputField : outputFields){
				ResultFeature resultFeature = outputField.getResultFeature();

				switch(resultFeature){
					case PROBABILITY:
						{
							outputField.setName(getDecisionFunctionField(outputField.getValue()));

							decisionFunctionFeatures.add(new ContinuousFeature(encoder, outputField));
						}
						break;
					default:
						break;
				}
			}

			models.add(model);

			// XXX
			if(categoricalLabel.size() == 2){
				SchemaUtil.checkSize(2, decisionFunctionFeatures);

				decisionFunctionFeatures = decisionFunctionFeatures.subList(1, 2);
			}
		}

		SchemaUtil.checkSize(calibrators.size(), decisionFunctionFeatures);

		RegressionModel calibratorModel;

		if(calibrators.size() == 1){
			SchemaUtil.checkSize(2, categoricalLabel);

			Calibrator calibrator = calibrators.get(0);
			Model featureModel = models.get(0);
			Feature feature = decisionFunctionFeatures.get(0);

			Feature calibratedFeature = calibrate(calibrator, featureModel, feature, encoder);

			calibratorModel = RegressionModelUtil.createBinaryLogisticClassification(Collections.singletonList(calibratedFeature), Collections.singletonList(1d), null, RegressionModel.NormalizationMethod.NONE, false, schema);
		} else

		if(calibrators.size() >= 3){
			SchemaUtil.checkSize(calibrators.size(), categoricalLabel);

			List<RegressionTable> regressionTables = new ArrayList<>();

			for(int i = 0; i < calibrators.size(); i++){
				Calibrator calibrator = calibrators.get(i);
				Model featureModel = models.size() == 1 ? models.get(0) : models.get(i);
				Feature feature = decisionFunctionFeatures.get(i);

				Feature calibratedFeature = calibrate(calibrator, featureModel, feature, encoder);

				RegressionTable regressionTable = RegressionModelUtil.createRegressionTable(Collections.singletonList(calibratedFeature), Collections.singletonList(1d), null)
					.setTargetCategory(categoricalLabel.getValue(i));

				regressionTables.add(regressionTable);
			}

			calibratorModel = new RegressionModel(MiningFunction.CLASSIFICATION, ModelUtil.createMiningSchema(categoricalLabel), regressionTables)
				.setNormalizationMethod(RegressionModel.NormalizationMethod.SIMPLEMAX);
		} else

		{
			throw new IllegalArgumentException();
		}

		encodePredictProbaOutput(calibratorModel, DataType.DOUBLE, categoricalLabel);

		models.add(calibratorModel);

		return MiningModelUtil.createModelChain(models, Segmentation.MissingPredictionTreatment.RETURN_MISSING);
	}

	@Override
	public List<?> getClasses(){
		return getClasses("classes");
	}

	public List<Calibrator> getCalibrators(){
		return getList("calibrators", Calibrator.class);
	}

	@Override
	public Classifier getEstimator(){
		return get("estimator", Classifier.class);
	}

	public String getMethod(){
		return getEnum("method", this::getString, Arrays.asList(CalibratedClassifier.METHOD_ISOTONIC, CalibratedClassifier.METHOD_SIGMOID));
	}

	private String getDecisionFunctionField(Object value){
		Object pmmlSegmentId = getPMMLSegmentId();

		if(value instanceof String){
			String name = (String)value;

			// XXX
			value = extractArguments(Estimator.FIELD_DECISION_FUNCTION, name);
		}

		List<?> args;

		if(pmmlSegmentId != null){
			args = Arrays.asList(pmmlSegmentId, value);
		} else

		{
			args = Arrays.asList(value);
		}

		return FieldNameUtil.create(Estimator.FIELD_DECISION_FUNCTION, args);
	}

	static
	private Feature calibrate(Calibrator calibrator, Model model, Feature feature, SkLearnEncoder encoder){
		encoder.export(model, feature.getName());

		Feature calibratedFeature = Iterables.getOnlyElement(calibrator.encodeFeatures(Collections.singletonList(feature), encoder));

		DerivedField derivedField = encoder.removeDerivedField(calibratedFeature.getName());

		OutputField outputField = new OutputField(derivedField.requireName(), derivedField.requireOpType(), derivedField.requireDataType())
			.setResultFeature(ResultFeature.TRANSFORMED_VALUE)
			.setExpression(derivedField.requireExpression())
			.setFinalResult(false);

		derivedField = encoder.createDerivedField(model, outputField, true);

		return new ContinuousFeature(encoder, derivedField);
	}

	private static final String METHOD_ISOTONIC = "isotonic";
	private static final String METHOD_SIGMOID = "sigmoid";

}