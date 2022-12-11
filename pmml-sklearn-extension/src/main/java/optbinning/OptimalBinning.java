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

import java.util.ArrayList;
import java.util.List;

import org.jpmml.python.ClassDictUtil;
import org.jpmml.python.PythonObject;

public class OptimalBinning extends PythonObject {

	public OptimalBinning(String module, String name){
		super(module, name);
	}

	public List<Double> getCategories(){
		List<Integer> numberOfEvents = getNumberOfEvents();
		List<Integer> numberOfNonEvents = getNumberOfNonEvents();

		ClassDictUtil.checkSize(numberOfEvents, numberOfNonEvents);

		double constant = Math.log((double)OptimalBinningUtil.sumExact(numberOfEvents) / (double)OptimalBinningUtil.sumExact(numberOfNonEvents));

		List<Double> result = new ArrayList<>();

		for(int i = 0; i < numberOfEvents.size(); i++){
			double eventRate = (double)numberOfEvents.get(i) / (double)Math.addExact(numberOfEvents.get(i), numberOfNonEvents.get(i));

			double woe = Math.log((1d / eventRate) - 1d) + constant;

			result.add(woe);
		}

		return result;
	}

	public String getDType(){
		return getString("dtype");
	}

	public List<Integer> getNumberOfEvents(){
		return getIntegerArray("_n_event");
	}

	public List<Integer> getNumberOfNonEvents(){
		return getIntegerArray("_n_nonevent");
	}

	public List<Number> getSplitsOptimal(){
		return getNumberArray("_splits_optimal");
	}
}