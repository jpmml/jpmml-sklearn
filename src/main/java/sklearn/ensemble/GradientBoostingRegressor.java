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
package sklearn.ensemble;

import java.util.List;

import org.dmg.pmml.DataField;
import org.dmg.pmml.DataType;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.FieldUsageType;
import org.dmg.pmml.MiningFunctionType;
import org.dmg.pmml.MiningModel;
import org.dmg.pmml.MiningSchema;
import org.dmg.pmml.MultipleModelMethodType;
import org.dmg.pmml.NumericPredictor;
import org.dmg.pmml.Output;
import org.dmg.pmml.RegressionModel;
import org.dmg.pmml.RegressionTable;
import org.dmg.pmml.Segment;
import org.dmg.pmml.Segmentation;
import org.dmg.pmml.True;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.sklearn.ClassDictUtil;
import sklearn.Regressor;
import sklearn.linear_model.RegressionModelUtil;
import sklearn.tree.DecisionTreeRegressor;
import sklearn.tree.TreeModelUtil;

public class GradientBoostingRegressor extends Regressor {

	public GradientBoostingRegressor(String module, String name){
		super(module, name);
	}

	@Override
	public DataType getDataType(){
		return DataType.FLOAT;
	}

	@Override
	public int getNumberOfFeatures(){
		return (Integer)get("n_features");
	}

	@Override
	public MiningModel encodeModel(List<DataField> dataFields){
		Segmentation segmentation = new Segmentation(MultipleModelMethodType.MODEL_CHAIN, null);

		FieldName decisionFunction = FieldName.create("decisionFunction");

		// Computes the value of the decision function
		{
			Output output = new Output()
				.addOutputFields(PMMLUtil.createPredictedField(decisionFunction));

			List<DecisionTreeRegressor> estimators = getEstimators();

			MiningModel miningModel = TreeModelUtil.encodeTreeModelEnsemble(estimators, null, MultipleModelMethodType.SUM, MiningFunctionType.REGRESSION, dataFields, false)
				.setOutput(output);

			Segment segment = new Segment()
				.setPredicate(new True())
				.setModel(miningModel);

			segmentation.addSegments(segment);
		}

		// Scales and shifts the the value of the decision function
		{
			Object init = getInit();

			if(!(init instanceof HasIntercept)){
				throw new IllegalArgumentException("The estimator object (" + ClassDictUtil.formatClass(init) + ") is not a BaseEstimator or is not a supported BaseEstimator subclass");
			}

			HasIntercept hasIntercept = (HasIntercept)init;

			Number learningRate = getLearningRate();

			DataField dataField = dataFields.get(0);

			MiningSchema miningSchema = new MiningSchema()
				.addMiningFields(PMMLUtil.createMiningField(dataField.getName(), FieldUsageType.TARGET))
				.addMiningFields(PMMLUtil.createMiningField(decisionFunction));

			NumericPredictor numericPredictor = new NumericPredictor(decisionFunction, learningRate.doubleValue());

			RegressionTable regressionTable = RegressionModelUtil.encodeRegressionTable(numericPredictor, hasIntercept.getIntercept());

			RegressionModel regressionModel = new RegressionModel(MiningFunctionType.REGRESSION, miningSchema, null)
				.addRegressionTables(regressionTable);

			Segment segment = new Segment()
				.setPredicate(new True())
				.setModel(regressionModel);

			segmentation.addSegments(segment);
		}

		MiningSchema miningSchema = PMMLUtil.createMiningSchema(dataFields);

		MiningModel miningModel = new MiningModel(MiningFunctionType.REGRESSION, miningSchema)
			.setSegmentation(segmentation);

		return miningModel;
	}

	public Object getInit(){
		return get("init_");
	}

	public Number getLearningRate(){
		return (Number)get("learning_rate");
	}

	public List<DecisionTreeRegressor> getEstimators(){
		return (List)ClassDictUtil.getArray(this, "estimators_");
	}
}