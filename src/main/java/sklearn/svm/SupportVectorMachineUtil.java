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
package sklearn.svm;

import java.util.List;

import com.google.common.base.Function;
import com.google.common.collect.Lists;
import org.dmg.pmml.support_vector_machine.Kernel;
import org.dmg.pmml.support_vector_machine.LinearKernel;
import org.dmg.pmml.support_vector_machine.PolynomialKernel;
import org.dmg.pmml.support_vector_machine.RadialBasisKernel;
import org.dmg.pmml.support_vector_machine.SigmoidKernel;
import org.jpmml.converter.ValueUtil;

public class SupportVectorMachineUtil {

	private SupportVectorMachineUtil(){
	}

	static
	public List<String> formatIds(List<Integer> values){
		Function<Integer, String> function = new Function<Integer, String>(){

			@Override
			public String apply(Integer value){
				return value.toString();
			}
		};

		return Lists.transform(values, function);
	}

	static
	public Kernel createKernel(String kernel, Integer degree, Double gamma, Double coef0){

		switch(kernel){
			case "linear":
				return new LinearKernel();
			case "poly":
				return new PolynomialKernel()
					.setGamma(gamma)
					.setCoef0(coef0)
					.setDegree(ValueUtil.asDouble(degree));
			case "rbf":
				return new RadialBasisKernel()
					.setGamma(gamma);
			case "sigmoid":
				return new SigmoidKernel()
					.setGamma(gamma)
					.setCoef0(coef0);
			default:
				throw new IllegalArgumentException(kernel);
		}
	}
}