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

import org.dmg.pmml.DataType;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.Model;
import org.dmg.pmml.OpType;
import org.dmg.pmml.regression.RegressionModel;
import org.dmg.pmml.regression.RegressionTable;
import org.jpmml.converter.CMatrixUtil;
import org.jpmml.converter.CategoricalLabel;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.Feature;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.PMMLEncoder;
import org.jpmml.converter.Schema;
import org.jpmml.converter.SchemaUtil;
import org.jpmml.converter.mining.MiningModelUtil;
import org.jpmml.converter.regression.RegressionModelUtil;
import org.jpmml.python.ClassDictUtil;
import org.jpmml.sklearn.SkLearnUtil;
import sklearn.linear_model.LinearClassifier;

public class LogisticRegression extends LinearClassifier {

	public LogisticRegression(String module, String name){
		super(module, name);
	}

	@Override
	public Model encodeModel(Schema schema){
		String multiClass = getMultiClass();

		if(("auto").equals(multiClass)){
			throw new IllegalArgumentException("Attribute \'" + ClassDictUtil.formatMember(this, "multi_class") + "\' must be explicitly set to the 'ovr' or 'multinomial' value");
		} else

		if(("multinomial").equals(multiClass)){
			return encodeMultinomialModel(schema);
		} else

		if(("ovr").equals(multiClass)){
			return encodeOvRModel(schema);
		} else

		{
			throw new IllegalArgumentException(multiClass);
		}
	}

	private Model encodeMultinomialModel(Schema schema){
		String sklearnVersion = getSkLearnVersion();
		int[] shape = getCoefShape();

		int numberOfClasses = shape[0];
		int numberOfFeatures = shape[1];

		List<? extends Number> coef = getCoef();
		List<? extends Number> intercept = getIntercept();

		PMMLEncoder encoder = schema.getEncoder();
		CategoricalLabel categoricalLabel = (CategoricalLabel)schema.getLabel();
		List<? extends Feature> features = schema.getFeatures();

		if(numberOfClasses == 1){
			SchemaUtil.checkSize(2, categoricalLabel);

			// See https://github.com/scikit-learn/scikit-learn/issues/9889
			boolean corrected = (sklearnVersion != null && SkLearnUtil.compareVersion(sklearnVersion, "0.20") >= 0);

			if(!corrected){
				return encodeOvRModel(schema);
			}

			Schema segmentSchema = schema.toRelabeledSchema(null);

			RegressionModel firstRegressionModel = RegressionModelUtil.createRegression(features, CMatrixUtil.getRow(coef, 1, numberOfFeatures, 0), intercept.get(0), null, segmentSchema)
				.setOutput(ModelUtil.createPredictedOutput(FieldName.create("decisionFunction"), OpType.CONTINUOUS, DataType.DOUBLE));

			Feature feature = new ContinuousFeature(encoder, FieldName.create("decisionFunction"), DataType.DOUBLE);

			RegressionTable passiveRegressionTable = RegressionModelUtil.createRegressionTable(Collections.singletonList(feature), Collections.singletonList(-1d), 0d)
				.setTargetCategory(categoricalLabel.getValue(0));

			RegressionTable activeRegressionTable = RegressionModelUtil.createRegressionTable(Collections.singletonList(feature), Collections.singletonList(1d), 0d)
				.setTargetCategory(categoricalLabel.getValue(1));

			List<RegressionTable> regressionTables = new ArrayList<>();
			regressionTables.add(passiveRegressionTable);
			regressionTables.add(activeRegressionTable);

			RegressionModel secondRegressionModel = new RegressionModel(MiningFunction.CLASSIFICATION, ModelUtil.createMiningSchema(categoricalLabel), regressionTables)
				.setNormalizationMethod(RegressionModel.NormalizationMethod.SOFTMAX)
				.setOutput(ModelUtil.createProbabilityOutput(DataType.DOUBLE, categoricalLabel));

			return MiningModelUtil.createModelChain(Arrays.asList(firstRegressionModel, secondRegressionModel));
		} else

		if(numberOfClasses >= 3){
			SchemaUtil.checkSize(numberOfClasses, categoricalLabel);

			List<RegressionTable> regressionTables = new ArrayList<>();

			for(int i = 0; i < categoricalLabel.size(); i++){
				RegressionTable regressionTable = RegressionModelUtil.createRegressionTable(features, CMatrixUtil.getRow(coef, numberOfClasses, numberOfFeatures, i), intercept.get(i))
					.setTargetCategory(categoricalLabel.getValue(i));

				regressionTables.add(regressionTable);
			}

			RegressionModel regressionModel = new RegressionModel(MiningFunction.CLASSIFICATION, ModelUtil.createMiningSchema(categoricalLabel), regressionTables)
				.setNormalizationMethod(RegressionModel.NormalizationMethod.SOFTMAX)
				.setOutput(ModelUtil.createProbabilityOutput(DataType.DOUBLE, categoricalLabel));

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
		String multiClass = getString("multi_class");

		// SkLearn 0.20
		if(("warn").equals(multiClass)){
			return "ovr";
		}

		return multiClass;
	}
}