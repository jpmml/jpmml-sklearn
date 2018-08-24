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
package sklearn;

import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

import com.google.common.collect.Iterables;
import org.dmg.pmml.DataType;
import org.jpmml.converter.ValueUtil;

public class TypeUtil {

	private TypeUtil(){
	}

	static
	public DataType getDataType(List<?> objects, DataType defaultDataType){
		Set<DataType> dataTypes = objects.stream()
			.map(ValueUtil::getDataType)
			.collect(Collectors.toSet());

		if(dataTypes.size() == 0){
			return defaultDataType;
		} // End if

		// A String value can be parsed to any other value
		if(dataTypes.size() > 1){
			dataTypes.remove(DataType.STRING);
		} // End if

		if(dataTypes.size() == 1){
			return Iterables.getOnlyElement(dataTypes);
		} else

		{
			throw new IllegalArgumentException("Expected all values to be of the same data type, got " + dataTypes.size() + " different data types (" + dataTypes + ")");
		}
	}
}