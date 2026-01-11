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
package sklearn.linear_model.logistic;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

import org.dmg.pmml.DataType;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.Model;
import org.dmg.pmml.OpType;
import org.dmg.pmml.mining.MiningModel;
import org.dmg.pmml.mining.Segmentation;
import org.dmg.pmml.regression.RegressionModel;
import org.dmg.pmml.regression.RegressionTable;
import org.jpmml.converter.CMatrixUtil;
import org.jpmml.converter.CategoricalLabel;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.ExceptionUtil;
import org.jpmml.converter.Feature;
import org.jpmml.converter.ModelEncoder;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.Schema;
import org.jpmml.converter.mining.MiningModelUtil;
import org.jpmml.converter.regression.RegressionModelUtil;
import org.jpmml.python.Attribute;
import org.jpmml.python.InvalidAttributeException;
import org.jpmml.python.MissingAttributeException;
import org.jpmml.python.PythonFormatterUtil;
import sklearn.Estimator;
import sklearn.VersionUtil;
import sklearn.linear_model.LinearClassifier;

public class LogisticRegression extends LinearClassifier {

	public LogisticRegression(String module, String name){
		super(module, name);
	}

	@Override
	public Model encodeModel(Schema schema){
		String sklearnVersion = getSkLearnVersion();
		String multiClass = getMultiClass();

		if(Objects.equals(LogisticRegression.MULTICLASS_AUTO, multiClass)){
			int[] shape = getCoefShape();
			String solver = getSolver();

			if(sklearnVersion != null && VersionUtil.compareVersion(sklearnVersion, "0.22") >= 0){
				multiClass = getAutoMultiClass(solver, shape);
			} else

			{
				throw new InvalidAttributeException("Attribute " + ExceptionUtil.formatName("multi_class") + " has unsupported value " + PythonFormatterUtil.formatValue(LogisticRegression.MULTICLASS_AUTO), new Attribute(this, "multi_class"))
					.setSolution(formatEnumChoice(Arrays.asList(LogisticRegression.MULTICLASS_MULTINOMIAL, LogisticRegression.MULTICLASS_OVR)));
			}
		} else

		if(Objects.equals(LogisticRegression.MULTICLASS_DEPRECATED, multiClass)){
			int[] shape = getCoefShape();

			if(sklearnVersion != null && VersionUtil.compareVersion(sklearnVersion, "1.5.0") >= 0){
				multiClass = getAutoMultiClass(null, shape);
			} else

			{
				throw new InvalidAttributeException("Attribute " + ExceptionUtil.formatName("multi_class") + " has unsupported value " + PythonFormatterUtil.formatValue(LogisticRegression.MULTICLASS_DEPRECATED), new Attribute(this, "multi_class"))
					.setSolution(formatEnumChoice(Arrays.asList(LogisticRegression.MULTICLASS_MULTINOMIAL, LogisticRegression.MULTICLASS_OVR)));
			}
		} else

		if(Objects.equals(null, multiClass)){
			int[] shape = getCoefShape();

			if(sklearnVersion == null || VersionUtil.compareVersion(sklearnVersion, "1.8.0") >= 0){
				multiClass = getAutoMultiClass(null, shape);
			}
		} // End if

		if(multiClass == null){
			throw new MissingAttributeException(new Attribute(this, "multi_class"));
		}

		switch(multiClass){
			case LogisticRegression.MULTICLASS_MULTINOMIAL:
				return encodeMultinomialModel(schema);
			case LogisticRegression.MULTICLASS_OVR:
				return encodeOvRModel(schema);
			default:
				throw new IllegalArgumentException(multiClass);
		}
	}

	private Model encodeMultinomialModel(Schema schema){
		String sklearnVersion = getSkLearnVersion();
		int[] shape = getCoefShape();

		int numberOfClasses = shape[0];
		int numberOfFeatures = shape[1];

		List<Number> coef = getCoef();
		List<Number> intercept = getIntercept();

		ModelEncoder encoder = schema.getEncoder();
		CategoricalLabel categoricalLabel = schema.requireCategoricalLabel();
		List<? extends Feature> features = schema.getFeatures();

		if(numberOfClasses == 1){
			categoricalLabel.expectCardinality(2);

			// See https://github.com/scikit-learn/scikit-learn/issues/9889
			boolean corrected = (sklearnVersion != null && VersionUtil.compareVersion(sklearnVersion, "0.20") >= 0);

			if(!corrected){
				return encodeOvRModel(schema);
			}

			Schema segmentSchema = schema.toRelabeledSchema(null);

			Model firstModel = RegressionModelUtil.createRegression(features, CMatrixUtil.getRow(coef, 1, numberOfFeatures, 0), intercept.get(0), null, segmentSchema)
				.setOutput(ModelUtil.createPredictedOutput(Estimator.FIELD_DECISION_FUNCTION, OpType.CONTINUOUS, DataType.DOUBLE));

			Feature feature = new ContinuousFeature(encoder, Estimator.FIELD_DECISION_FUNCTION, DataType.DOUBLE);

			RegressionTable passiveRegressionTable = RegressionModelUtil.createRegressionTable(Collections.singletonList(feature), Collections.singletonList(-1d), 0d)
				.setTargetCategory(categoricalLabel.getValue(0));

			RegressionTable activeRegressionTable = RegressionModelUtil.createRegressionTable(Collections.singletonList(feature), Collections.singletonList(1d), 0d)
				.setTargetCategory(categoricalLabel.getValue(1));

			List<RegressionTable> regressionTables = new ArrayList<>();
			regressionTables.add(passiveRegressionTable);
			regressionTables.add(activeRegressionTable);

			Model secondModel = new RegressionModel(MiningFunction.CLASSIFICATION, ModelUtil.createMiningSchema(categoricalLabel), regressionTables)
				.setNormalizationMethod(RegressionModel.NormalizationMethod.SOFTMAX);

			MiningModel miningModel = MiningModelUtil.createModelChain(Arrays.asList(firstModel, secondModel), Segmentation.MissingPredictionTreatment.RETURN_MISSING);

			encodePredictProbaOutput(miningModel, DataType.DOUBLE, categoricalLabel);

			return miningModel;
		} else

		if(numberOfClasses >= 3){
			categoricalLabel.expectCardinality(numberOfClasses);

			List<RegressionTable> regressionTables = new ArrayList<>();

			for(int i = 0; i < categoricalLabel.size(); i++){
				RegressionTable regressionTable = RegressionModelUtil.createRegressionTable(features, CMatrixUtil.getRow(coef, numberOfClasses, numberOfFeatures, i), intercept.get(i))
					.setTargetCategory(categoricalLabel.getValue(i));

				regressionTables.add(regressionTable);
			}

			RegressionModel regressionModel = new RegressionModel(MiningFunction.CLASSIFICATION, ModelUtil.createMiningSchema(categoricalLabel), regressionTables)
				.setNormalizationMethod(RegressionModel.NormalizationMethod.SOFTMAX);

			encodePredictProbaOutput(regressionModel, DataType.DOUBLE, categoricalLabel);

			return regressionModel;
		} else

		{
			throw new IllegalArgumentException();
		}
	}

	private Model encodeOvRModel(Schema schema){
		return super.encodeModel(schema);
	}

	public String getMultiClass(){

		// SkLearn 1.8.0+
		if(!hasattr("multi_class")){
			return null;
		}

		String multiClass = getEnum("multi_class", this::getString, Arrays.asList(LogisticRegression.MULTICLASS_AUTO, LogisticRegression.MULTICLASS_DEPRECATED, LogisticRegression.MULTICLASS_MULTINOMIAL, LogisticRegression.MULTICLASS_OVR, LogisticRegression.MULTICLASS_WARN));

		// SkLearn 0.20
		if(Objects.equals(LogisticRegression.MULTICLASS_WARN, multiClass)){
			multiClass = LogisticRegression.MULTICLASS_OVR;
		}

		return multiClass;
	}

	public String getSolver(){
		return getString("solver");
	}

	static
	private String getAutoMultiClass(String solver, int[] shape){
		int numberOfClasses = shape[0];
		int numberOfFeatures = shape[1];

		if(Objects.equals(LogisticRegression.SOLVER_LIBLINEAR, solver)){
			return LogisticRegression.MULTICLASS_OVR;
		} // End if

		if(numberOfClasses == 1){
			return LogisticRegression.MULTICLASS_OVR;
		} else

		{
			return LogisticRegression.MULTICLASS_MULTINOMIAL;
		}
	}

	private static final String MULTICLASS_AUTO = "auto";
	private static final String MULTICLASS_DEPRECATED = "deprecated";
	private static final String MULTICLASS_MULTINOMIAL = "multinomial";
	private static final String MULTICLASS_OVR = "ovr";
	private static final String MULTICLASS_WARN = "warn";

	private static final String SOLVER_LIBLINEAR = "liblinear";
}