/*
 * Copyright (c) 2022 Villu Ruusmann
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
package optbinning;

import java.util.Collection;
import java.util.List;

import org.jpmml.python.PythonObject;

abstract
public class Binning extends PythonObject {

	public Binning(String module, String name){
		super(module, name);
	}

	abstract
	public List<? extends Number> getCategories();

	public String getDType(){
		return getString("dtype");
	}

	public List<Number> getSplitsOptimal(){
		return getNumberArray("_splits_optimal");
	}

	static
	protected int sum(Collection<Integer> values){
		int result = 0;

		for(Integer value : values){
			result = Math.addExact(result, value);
		}

		return result;
	}
}