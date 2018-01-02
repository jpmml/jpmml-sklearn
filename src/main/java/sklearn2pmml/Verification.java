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
package sklearn2pmml;

import java.util.List;

import org.jpmml.sklearn.PyClassDict;

public class Verification extends PyClassDict {

	public Verification(String module, String name){
		super(module, name);
	}

	public boolean hasProbabilityValues(){
		return containsKey("probability_values");
	}

	public List<?> getActiveValues(){
		return getArray("active_values");
	}

	public int[] getActiveValuesShape(){
		return getArrayShape("active_values", 2);
	}

	public List<? extends Number> getProbabilityValues(){
		return getArray("probability_values", Number.class);
	}

	public int[] getProbabilityValuesShape(){
		return getArrayShape("probability_values", 2);
	}

	public List<?> getTargetValues(){
		return getArray("target_values");
	}

	public int[] getTargetValuesShape(){
		return getArrayShape("target_values");
	}

	public Number getPrecision(){
		return (Number)get("precision");
	}

	public Number getZeroThreshold(){
		return (Number)get("zeroThreshold");
	}
}