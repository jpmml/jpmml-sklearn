/*
 * Copyright (c) 2017 Villu Ruusmann
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
package org.jpmml.sklearn;

import org.dmg.pmml.DataType;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.OpType;
import org.jpmml.converter.Transformation;

abstract
public class OutlierTransformation implements Transformation {

	@Override
	public FieldName getName(FieldName name){
		return FieldName.create("outlier");
	}

	@Override
	public OpType getOpType(OpType opType){
		return OpType.CATEGORICAL;
	}

	@Override
	public DataType getDataType(DataType dataType){
		return DataType.BOOLEAN;
	}

	@Override
	public boolean isFinalResult(){
		return true;
	}
}