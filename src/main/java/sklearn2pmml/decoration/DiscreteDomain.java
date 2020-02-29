/*
 * Copyright (c) 2020 Villu Ruusmann
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
package sklearn2pmml.decoration;

import java.util.List;

import org.dmg.pmml.DataType;
import org.dmg.pmml.DiscrStats;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.converter.TypeUtil;
import org.jpmml.converter.ValueUtil;
import org.jpmml.sklearn.ClassDictUtil;
import sklearn.TransformerUtil;

abstract
public class DiscreteDomain extends Domain {

	public DiscreteDomain(String module, String name){
		super(module, name);
	}

	@Override
	public DataType getDataType(){
		Object dtype = getDType();
		Boolean withData = getWithData();

		if(dtype != null){
			return TransformerUtil.getDataType(dtype);
		} // End if

		if(withData){
			List<?> data = getData();

			return TypeUtil.getDataType(data, DataType.STRING);
		}

		return DataType.STRING;
	}

	public List<?> getData(){
		return getArray("data_");
	}

	public Object[] getDiscrStats(){
		return getTuple("discr_stats_");
	}

	static
	public DiscrStats createDiscrStats(Object[] objects){
		List<Object> values = (List)asArray(objects[0]);
		List<Integer> counts = ValueUtil.asIntegers((List)asArray(objects[1]));

		ClassDictUtil.checkSize(values, counts);

		DiscrStats discrStats = new DiscrStats()
			.addArrays(PMMLUtil.createStringArray(values), PMMLUtil.createIntArray(counts));

		return discrStats;
	}
}