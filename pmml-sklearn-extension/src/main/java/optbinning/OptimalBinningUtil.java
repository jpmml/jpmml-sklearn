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
import java.util.Iterator;
import java.util.List;

public class OptimalBinningUtil {

	private OptimalBinningUtil(){
	}

	static
	public void checkIncreasingOrder(List<Number> bins){
		Iterator<Number> it = bins.iterator();

		Number prevBin = it.next();

		while(it.hasNext()){
			Number bin = it.next();

			if(bin.doubleValue() <= prevBin.doubleValue()){
				throw new IllegalArgumentException();
			}

			prevBin = bin;
		}
	}

	static
	public int sumExact(Collection<Integer> values){
		int result = 0;

		for(Integer value : values){
			result = Math.addExact(result, value);
		}

		return result;
	}
}