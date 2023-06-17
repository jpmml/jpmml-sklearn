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
import sklearn.HasEstimator;
import sklearn.linear_model.LinearClassifier;

public class CalibratedClassifier extends Classifier implements HasEstimator<Classifier> {

	public CalibratedClassifier(String module, String name){
		super(module, name);
	}

	@Override
	public Model encodeModel(Schema schema){
		List<? extends Calibrator> calibrators = getCalibrators();
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

		Model model = estimator.encode(schema);

		List<Model> models = new ArrayList<>();

		List<Feature> decisionFunctionFeatures = new ArrayList<>();

		if(estimator instanceof LinearClassifier){

			if(model instanceof MiningModel){
				MiningModel miningModel = (MiningModel)model;

				Segmentation segmentation = miningModel.requireSegmentation();

				List<Segment> segments = segmentation.requireSegments();
				for(Segment segment : segments){
					RegressionModel decisionFunctionModel = (RegressionModel)segment.requireModel();

					if(decisionFunctionModel.requireMiningFunction() != MiningFunction.REGRESSION){
						continue;
					}

					Output output = decisionFunctionModel.getOutput();
					if(output == null){
						throw new IllegalArgumentException();
					}

					OutputField outputField = Iterables.getOnlyElement(output.getOutputFields());

					decisionFunctionModel.setNormalizationMethod(RegressionModel.NormalizationMethod.NONE);

					models.add(decisionFunctionModel);

					decisionFunctionFeatures.add(new ContinuousFeature(encoder, outputField));
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
					OutputField outputField = ModelUtil.createPredictedField(FieldNameUtil.create(Estimator.FIELD_DECISION_FUNCTION, regressionTable.requireTargetCategory()), OpType.CONTINUOUS, DataType.DOUBLE)
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
			Output output = model.getOutput();
			if(output == null){
				throw new IllegalArgumentException();
			}

			List<OutputField> outputFields = output.getOutputFields();
			for(OutputField outputField : outputFields){
				ResultFeature resultFeature = outputField.getResultFeature();

				switch(resultFeature){
					case PROBABILITY:
						{
							// XXX
							outputField.setName(FieldNameUtil.create(Estimator.FIELD_DECISION_FUNCTION, outputField.getValue()));

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
		return getListLike("classes");
	}

	public List<? extends Calibrator> getCalibrators(){
		return getList("calibrators", Calibrator.class);
	}

	@Override
	public Classifier getEstimator(){
		return get("estimator", Classifier.class);
	}

	public String getMethod(){
		return getString("method");
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
}