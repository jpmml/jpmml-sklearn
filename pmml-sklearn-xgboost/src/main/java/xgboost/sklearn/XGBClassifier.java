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
import org.dmg.pmml.PMML;
import org.dmg.pmml.mining.MiningModel;
import org.jpmml.converter.Label;
import org.jpmml.converter.LabelUtil;
import org.jpmml.converter.Schema;
import org.jpmml.sklearn.Encodable;
import org.jpmml.sklearn.SkLearnException;
import org.jpmml.xgboost.Classification;
import org.jpmml.xgboost.HasXGBoostOptions;
import org.jpmml.xgboost.ObjFunction;
import sklearn.LabelEncoderClassifier;
import sklearn.SkLearnFields;
import sklearn2pmml.SkLearn2PMMLFields;

public class XGBClassifier extends LabelEncoderClassifier implements HasBooster, HasXGBoostOptions, Encodable {

	public XGBClassifier(String module, String name){
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
	public List<?> getClasses(){

		// XGBoost 0.4 through 1.7
		if(hasattr("_le") || hasattr(SkLearnFields.CLASSES) || hasattr(SkLearn2PMMLFields.PMML_CLASSES)){
			return super.getClasses();
		}

		// XGBoost 2.0+
		int numClasses = getNumberOfClasses();

		return LabelUtil.createTargetCategories(numClasses);
	}

	@Override
	public void checkLabel(Label label){
		ObjFunction objFunction = BoosterUtil.getObjFunction(this);

		super.checkLabel(label);

		if((objFunction != null) && !(objFunction instanceof Classification)){
			throw new SkLearnException("Expected a classification-type objective function, got '" + objFunction.getName() + "'");
		}
	}

	@Override
	public MiningModel encodeModel(Schema schema){
		return BoosterUtil.encodeModel(this, schema);
	}

	@Override
	public PMML encodePMML(){
		return BoosterUtil.encodePMML(this);
	}

	@Override
	public Booster getBooster(){
		return get("_Booster", Booster.class);
	}

	public int getNumberOfClasses(){
		return getInteger(SkLearnFields.N_CLASSES);
	}
}