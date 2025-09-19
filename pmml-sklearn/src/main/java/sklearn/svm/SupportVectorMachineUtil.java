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
	public Kernel createKernel(HasLibSVMKernel hasLibSVMKernel){
		String kernel = hasLibSVMKernel.getKernel();

		switch(kernel){
			case LibSVMConstants.KERNEL_LINEAR:
				{
					return new LinearKernel();
				}
			case LibSVMConstants.KERNEL_POLY:
				{
					Integer degree = hasLibSVMKernel.getDegree();
					Number gamma = hasLibSVMKernel.getGamma();
					Number coef0 = hasLibSVMKernel.getCoef0();

					return new PolynomialKernel()
						.setGamma(gamma)
						.setCoef0(coef0)
						.setDegree(degree);
				}
			case LibSVMConstants.KERNEL_RBF:
				{
					Number gamma = hasLibSVMKernel.getGamma();

					return new RadialBasisKernel()
						.setGamma(gamma);
				}
			case LibSVMConstants.KERNEL_SIGMOID:
				{
					Number gamma = hasLibSVMKernel.getGamma();
					Number coef0 = hasLibSVMKernel.getCoef0();

					return new SigmoidKernel()
						.setGamma(gamma)
						.setCoef0(coef0);
				}
			default:
				throw new IllegalArgumentException(kernel);
		}
	}
}