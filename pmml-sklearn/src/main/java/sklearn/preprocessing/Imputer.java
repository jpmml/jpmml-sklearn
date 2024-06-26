/*
 * Copyright (c) 2018 Villu Ruusmann
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
package sklearn.preprocessing;

import java.util.List;
import java.util.Objects;

import sklearn.impute.SimpleImputer;

public class Imputer extends SimpleImputer {

	public Imputer(String module, String name){
		super(module, name);
	}

	@Override
	public Object getMissingValues(){
		Object missingValues = super.getMissingValues();

		if(Objects.equals("NaN", missingValues)){
			missingValues = null;
		}

		return missingValues;
	}

	@Override
	public List<Object> getStatistics(){
		return (List)getNumberArray("statistics_");
	}
}