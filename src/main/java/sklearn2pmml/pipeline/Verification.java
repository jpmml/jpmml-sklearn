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
package sklearn2pmml.pipeline;

import java.util.List;

import com.google.common.base.Function;
import com.google.common.collect.Lists;
import net.razorvine.pickle.IObjectConstructor;
import org.jpmml.python.PythonObject;

public class Verification extends PythonObject {

	public Verification(String module, String name){
		super(module, name);
	}

	public boolean hasProbabilityValues(){
		return containsKey("probability_values");
	}

	public List<?> getActiveValues(){
		List<?> values = getArray("active_values");

		// XXX
		Function<Object, Object> function = new Function<Object, Object>(){

			@Override
			public Object apply(Object object){

				if(object instanceof IObjectConstructor){
					IObjectConstructor objectConstructor = (IObjectConstructor)object;

					return objectConstructor.construct(new Object[0]);
				}

				return object;
			}
		};

		return Lists.transform(values, function);
	}

	public int[] getActiveValuesShape(){
		int[] shape = getArrayShape("active_values");

		if(shape.length == 1){
			return new int[]{shape[0], 1};
		}

		return getArrayShape("active_values", 2);
	}

	public List<? extends Number> getProbabilityValues(){
		return getNumberArray("probability_values");
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
		return getNumber("precision");
	}

	public Number getZeroThreshold(){
		return getNumber("zeroThreshold");
	}
}