/*
 * Copyright (c) 2016 Villu Ruusmann
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
package xgboost.sklearn;

import java.util.List;

import org.dmg.pmml.DataType;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.MiningModel;
import org.jpmml.converter.FeatureSchema;
import org.jpmml.converter.Schema;
import org.jpmml.xgboost.FeatureMap;
import org.jpmml.xgboost.GBTree;
import org.jpmml.xgboost.Learner;
import sklearn.Regressor;

public class XGBRegressor extends Regressor {

	public XGBRegressor(String module, String name){
		super(module, name);
	}

	@Override
	public int getNumberOfFeatures(){
		Booster booster = getBooster();

		Learner learner = booster.getLearner();

		return learner.getNumFeatures();
	}

	@Override
	public DataType getDataType(){
		return DataType.FLOAT;
	}

	@Override
	public MiningModel encodeModel(Schema schema){
		Booster booster = getBooster();

		Learner learner = booster.getLearner();

		FieldName targetField = schema.getTargetField();
		List<String> targetCategories = schema.getTargetCategories();

		FeatureMap featureMap = new FeatureMap();

		List<FieldName> activeFields = schema.getActiveFields();
		for(int i = 0; i < activeFields.size(); i++){
			FieldName activeField = activeFields.get(i);

			featureMap.load(String.valueOf(i), activeField.getValue(), "q");
		}

		FeatureSchema featureSchema = featureMap.createSchema(targetField, targetCategories);

		GBTree gbt = learner.getGBTree();

		MiningModel miningModel = gbt.encodeMiningModel(learner.getObj(), learner.getBaseScore(), featureSchema);

		return miningModel;
	}

	public Booster getBooster(){
		return (Booster)get("_Booster");
	}
}