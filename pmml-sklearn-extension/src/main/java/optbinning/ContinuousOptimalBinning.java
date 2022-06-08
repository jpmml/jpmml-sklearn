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

public class ContinuousOptimalBinning extends Binning {

	public ContinuousOptimalBinning(String module, String name){
		super(module, name);
	}

	@Override
	public List<Double> getCategories(){
		List<Integer> numberOfRecords = getNumberOfRecords();
		List<Number> sums = getSums();

		ClassDictUtil.checkSize(numberOfRecords, sums);

		List<Double> result = new ArrayList<>();

		for(int i = 0; i < numberOfRecords.size(); i++){
			double mean = (sums.get(i)).doubleValue() / (double)numberOfRecords.get(i);

			result.add(mean);
		}

		return result;
	}

	public List<Integer> getNumberOfRecords(){
		return getIntegerArray("_n_records");
	}

	public List<Number> getSums(){
		return getNumberArray("_sums");
	}
}