/*
 * Copyright (c) 2015 Villu Ruusmann
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
package sklearn.linear_model.stochastic_gradient;

import sklearn.linear_model.BaseLinearClassifier;

public class SGDClassifier extends BaseLinearClassifier {

	public SGDClassifier(String module, String name){
		super(module, name);
	}

	@Override
	public boolean hasProbabilityDistribution(){
		LossFunction lossFunction = getLossFunction();

		if(lossFunction instanceof Log){
			return true;
		}

		return false;
	}

	public String getLoss(){
		return (String)get("loss");
	}

	public LossFunction getLossFunction(){

		// SkLearn 0.18
		if(containsKey("loss_function")){
			return get("loss_function", LossFunction.class);
		} else

		// SkLearn 0.19+
		{
			return get("loss_function_", LossFunction.class);
		}
	}
}