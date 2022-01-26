/*
 * Copyright (c) 2021 Villu Ruusmann
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
package category_encoders;

public class LeaveOneOutEncoder extends MeanEncoder {

	public LeaveOneOutEncoder(String module, String name){
		super(module, name);
	}

	@Override
	public String functionName(){
		return "loo";
	}

	@Override
	public MeanFunction createFunction(){
		Double mean = getMean();

		MeanFunction function = new MeanFunction(){

			@Override
			public Double apply(Double sum, Integer count){

				if(count > 1){
					return (sum / count);
				} else

				{
					return mean;
				}
			}
		};

		return function;
	}
}