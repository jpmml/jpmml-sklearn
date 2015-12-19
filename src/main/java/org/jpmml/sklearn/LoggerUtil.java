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
package org.jpmml.sklearn;

import java.util.ArrayList;
import java.util.List;

import org.dmg.pmml.FieldName;

public class LoggerUtil {

	private LoggerUtil(){
	}

	static
	public String formatNameList(List<FieldName> names){
	
		if(names.size() <= 10){
			return names.toString();
		}
	
		List<FieldName> result = new ArrayList<>();
		result.addAll(names.subList(0, 2));
		result.add(new FieldName("..."));
		result.addAll(names.subList(names.size() - 2, names.size()));
	
		return result.toString();
	}
}