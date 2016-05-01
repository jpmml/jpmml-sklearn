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
package sklearn.ensemble.gradient_boosting;

import java.util.ArrayList;
import java.util.List;

import org.dmg.pmml.FieldName;
import org.dmg.pmml.FieldUsageType;
import org.dmg.pmml.MiningField;
import org.dmg.pmml.MiningFunctionType;
import org.dmg.pmml.MiningModel;
import org.dmg.pmml.MiningSchema;
import org.dmg.pmml.Model;
import org.dmg.pmml.MultipleModelMethodType;
import org.dmg.pmml.NumericPredictor;
import org.dmg.pmml.Output;
import org.dmg.pmml.RegressionModel;
import org.dmg.pmml.RegressionTable;
import org.dmg.pmml.Segmentation;
import org.dmg.pmml.TreeModel;
import org.jpmml.converter.MiningModelUtil;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.Schema;
import org.jpmml.converter.ValueUtil;
import sklearn.linear_model.RegressionModelUtil;
import sklearn.tree.DecisionTreeRegressor;
import sklearn.tree.TreeModelUtil;

public class GradientBoostingUtil {

	private GradientBoostingUtil(){
	}

	static
	public MiningModel encodeGradientBoosting(List<DecisionTreeRegressor> regressors, Number initialPrediction, Number learningRate, Schema schema){
		List<Model> models = new ArrayList<>();

		FieldName sumField = FieldName.create("sum");

		{
			List<TreeModel> treeModels = TreeModelUtil.encodeTreeModelSegmentation(regressors, MiningFunctionType.REGRESSION, schema);

			Segmentation segmentation = MiningModelUtil.createSegmentation(MultipleModelMethodType.SUM, treeModels);

			Output output = new Output()
				.addOutputFields(ModelUtil.createPredictedField(sumField));

			MiningSchema miningSchema = ModelUtil.createMiningSchema(null, schema.getActiveFields());

			MiningModel miningModel = new MiningModel(MiningFunctionType.REGRESSION, miningSchema)
				.setSegmentation(segmentation)
				.setOutput(output);

			models.add(miningModel);
		} // End block

		{
			MiningField miningField = ModelUtil.createMiningField(sumField);

			NumericPredictor numericPredictor = new NumericPredictor(miningField.getName(), ValueUtil.asDouble(learningRate));

			RegressionTable regressionTable = RegressionModelUtil.encodeRegressionTable(numericPredictor, initialPrediction);

			MiningSchema miningSchema = new MiningSchema();

			FieldName targetField = schema.getTargetField();
			if(targetField != null){
				miningSchema.addMiningFields(ModelUtil.createMiningField(targetField, FieldUsageType.TARGET));
			}

			miningSchema.addMiningFields(miningField);

			RegressionModel regressionModel = new RegressionModel(MiningFunctionType.REGRESSION, miningSchema, null)
				.addRegressionTables(regressionTable);

			models.add(regressionModel);
		}

		Segmentation segmentation = MiningModelUtil.createSegmentation(MultipleModelMethodType.MODEL_CHAIN, models);

		MiningSchema miningSchema = ModelUtil.createMiningSchema(schema);

		MiningModel miningModel = new MiningModel(MiningFunctionType.REGRESSION, miningSchema)
			.setSegmentation(segmentation);

		return miningModel;
	}
}