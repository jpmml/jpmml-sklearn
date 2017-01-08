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
import org.dmg.pmml.mining.MiningModel;
import org.jpmml.converter.Schema;
import org.jpmml.sklearn.ClassDictUtil;
import sklearn.Classifier;
import sklearn.preprocessing.LabelEncoder;

public class XGBClassifier extends Classifier implements HasBooster {

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
		LabelEncoder labelEncoder = getLabelEncoder();

		return labelEncoder.getClasses();
	}

	@Override
	public MiningModel encodeModel(Schema schema){
		return BoosterUtil.encodeBooster(this, schema);
	}

	@Override
	public Booster getBooster(){
		return (Booster)get("_Booster");
	}

	public LabelEncoder getLabelEncoder(){
		Object object = get("_le");

		try {
			if(object == null){
				throw new NullPointerException();
			}

			return (LabelEncoder)object;
		} catch(RuntimeException re){
			throw new IllegalArgumentException("The label encoder object (" + ClassDictUtil.formatClass(object) + ") is not a LabelEncoder", re);
		}
	}
}