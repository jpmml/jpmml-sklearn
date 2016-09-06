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

import java.util.ArrayList;
import java.util.List;

import org.dmg.pmml.DataType;
import org.dmg.pmml.mining.MiningModel;
import org.jpmml.converter.Schema;
import org.jpmml.sklearn.HasArray;
import sklearn.Classifier;

public class XGBClassifier extends Classifier implements HasBooster {

	public XGBClassifier(String module, String name){
		super(module, name);
	}

	@Override
	public int getNumberOfFeatures(){
		return BoosterUtil.getNumberOfFeatures(this);
	}

	@Override
	public boolean requiresContinuousInput(){
		return false;
	}

	@Override
	public DataType getDataType(){
		return DataType.FLOAT;
	}

	@Override
	public List<?> getClasses(){
		List<Object> result = new ArrayList<>();

		List<?> values = (List<?>)get("classes_");
		for(Object value : values){

			if(value instanceof HasArray){
				HasArray hasArray = (HasArray)value;

				result.addAll(hasArray.getArrayContent());
			} else

			{
				result.add(value);
			}
		}

		return result;
	}

	@Override
	public MiningModel encodeModel(Schema schema){
		return BoosterUtil.encodeBooster(this, schema);
	}

	@Override
	public Booster getBooster(){
		return (Booster)get("_Booster");
	}
}