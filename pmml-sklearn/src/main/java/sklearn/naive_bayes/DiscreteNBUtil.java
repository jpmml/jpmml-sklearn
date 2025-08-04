/*
 * Copyright (c) 2025 Villu Ruusmann
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
package sklearn.naive_bayes;

import java.util.List;

import com.google.common.base.Function;
import com.google.common.collect.Lists;
import org.dmg.pmml.naive_bayes.PairCounts;
import org.dmg.pmml.naive_bayes.TargetValueCount;
import org.dmg.pmml.naive_bayes.TargetValueCounts;

public class DiscreteNBUtil {

	private DiscreteNBUtil(){
	}

	static
	public PairCounts encodePairCounts(Object value, List<?> values, Number alpha, List<? extends Number> counts){

		if(alpha.doubleValue() != 0d){
			Function<Number, Number> function = new Function<Number, Number>(){

				@Override
				public Number apply(Number count){
					return count.doubleValue() + alpha.doubleValue();
				}
			};

			counts = Lists.transform(counts, function);
		}

		PairCounts pairCounts = new PairCounts()
			.setValue(value)
			.setTargetValueCounts(encodeTargetValueCounts(values, counts));

		return pairCounts;
	}

	static
	public TargetValueCounts encodeTargetValueCounts(List<?> values, List<? extends Number> counts){
		TargetValueCounts targetValueCounts = new TargetValueCounts();

		for(int i = 0; i < values.size(); i++){
			Object value = values.get(i);
			Number count = counts.get(i);

			TargetValueCount targetValueCount = new TargetValueCount(value, count);

			targetValueCounts.addTargetValueCounts(targetValueCount);
		}

		return targetValueCounts;
	}
}