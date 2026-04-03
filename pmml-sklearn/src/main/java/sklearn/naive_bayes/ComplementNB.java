/*
 * Copyright (c) 2026 Villu Ruusmann
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
import org.jpmml.converter.ValueUtil;

public class ComplementNB extends ContinuousNB {

	public ComplementNB(String module, String name){
		super(module, name);
	}

	@Override
	protected List<Number> getCoefficients(List<Number> featureLogProb, int index, int numberOfClasses, int numberOfFeatures){

		if(numberOfClasses == 2){
			int complementIndex = (index == 0) ? 1 : 0;

			Function<Number, Number> function = new Function<Number, Number>(){

				@Override
				public Number apply(Number value){
					return (Number)ValueUtil.toNegative(value);
				}
			};

			return super.getCoefficients(Lists.transform(featureLogProb, function), complementIndex, numberOfClasses, numberOfFeatures);
		} else

		{
			return super.getCoefficients(featureLogProb, index, numberOfClasses, numberOfFeatures);
		}
	}

	@Override
	protected Number getIntercept(List<Number> classLogPrior, int index){
		return null;
	}
}