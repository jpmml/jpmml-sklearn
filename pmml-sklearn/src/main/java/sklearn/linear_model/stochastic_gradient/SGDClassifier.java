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

import org.jpmml.python.PythonObject;
import sklearn.linear_model.LinearClassifier;
import sklearn.loss.CyLossFunction;

public class SGDClassifier extends LinearClassifier {

	public SGDClassifier(String module, String name){
		super(module, name);
	}

	@Override
	public boolean hasProbabilityDistribution(){
		PythonObject lossFunction = getLossFunction();

		if(lossFunction instanceof Log){
			return true;
		} else

		if(lossFunction instanceof CyLossFunction){
			String pythonName = lossFunction.getPythonName();

			// XXX
			switch(pythonName){
				case "CyHalfBinomialLoss":
					return true;
				default:
					return false;
			}
		}

		return false;
	}

	public PythonObject getLossFunction(){

		// SkLearn 0.18
		if(hasattr("loss_function")){
			return get("loss_function", LossFunction.class);
		} else

		// SkLearn 0.19+
		if(hasattr("loss_function_")){
			return get("loss_function_", LossFunction.class);
		}

		Object lossFunction = get("_loss_function_");

		// SkLearn 1.4.0+
		if(lossFunction instanceof LossFunction){
			return get("_loss_function_", LossFunction.class);
		} else

		// SkLearn 1.6.0+
		{
			return get("_loss_function_", CyLossFunction.class);
		}
	}
}