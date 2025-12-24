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

import org.dmg.pmml.DataField;
import org.dmg.pmml.DataType;
import org.dmg.pmml.Model;
import org.dmg.pmml.OpType;
import org.dmg.pmml.PMML;
import org.dmg.pmml.mining.MiningModel;
import org.jpmml.converter.ContinuousLabel;
import org.jpmml.converter.Label;
import org.jpmml.converter.ScalarLabel;
import org.jpmml.converter.Schema;
import org.jpmml.sklearn.Encodable;
import org.jpmml.sklearn.SkLearnEncoder;
import org.jpmml.sklearn.SkLearnException;
import org.jpmml.xgboost.HasXGBoostOptions;
import org.jpmml.xgboost.ObjFunction;
import org.jpmml.xgboost.Regression;
import sklearn.Regressor;

public class XGBRegressor extends Regressor implements HasBooster, HasXGBoostOptions, Encodable {

	public XGBRegressor(String module, String name){
		super(module, name);
	}

	@Override
	public int getNumberOfFeatures(){
		return BoosterUtil.getNumberOfFeatures(this);
	}

	@Override
	public DataType getDataType(){
		return DataType.FLOAT;
	}

	@Override
	public void checkLabel(Label label){
		ObjFunction objFunction = BoosterUtil.getObjFunction(this);

		super.checkLabel(label);

		if((objFunction != null) && !(objFunction instanceof Regression)){
			throw new SkLearnException("Expected a regression-type objective function, got \'" + objFunction.getName() + "\'");
		}
	}

	@Override
	protected ScalarLabel encodeLabel(String name, SkLearnEncoder encoder){
		DataField dataField = encoder.createDataField(name, OpType.CONTINUOUS, DataType.FLOAT);

		return new ContinuousLabel(dataField);
	}

	@Override
	public MiningModel encodeModel(Schema schema){
		return BoosterUtil.encodeModel(this, schema);
	}

	@Override
	public Schema configureSchema(Schema schema){
		return BoosterUtil.configureSchema(this, schema);
	}

	@Override
	public MiningModel configureModel(Model model){
		return BoosterUtil.configureModel(this, (MiningModel)model);
	}

	@Override
	public PMML encodePMML(){
		return BoosterUtil.encodePMML(this);
	}

	@Override
	public Booster getBooster(){
		return get("_Booster", Booster.class);
	}
}