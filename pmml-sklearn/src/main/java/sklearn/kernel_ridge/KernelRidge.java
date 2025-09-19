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
package sklearn.kernel_ridge;

import java.util.AbstractList;
import java.util.List;

import org.dmg.pmml.support_vector_machine.Kernel;
import org.dmg.pmml.support_vector_machine.SupportVectorMachineModel;
import org.jpmml.converter.CMatrix;
import org.jpmml.converter.Feature;
import org.jpmml.converter.Schema;
import org.jpmml.converter.support_vector_machine.LibSVMUtil;
import sklearn.SkLearnRegressor;
import sklearn.svm.LibSVMConstants;
import sklearn.svm.SupportVectorMachineUtil;

public class KernelRidge extends SkLearnRegressor implements LibSVMConstants {

	public KernelRidge(String module, String name){
		super(module, name);
	}

	@Override
	public SupportVectorMachineModel encodeModel(Schema schema){
		List<Number> dualCoef = getDualCoef();
		List<Number> xFit = getXFit();

		List<? extends Feature> features = schema.getFeatures();

		int numberOfFeatures = features.size();
		int numberOfVectors = xFit.size() / numberOfFeatures;

		Kernel kernel = SupportVectorMachineUtil.createKernel(getKernel(), getDegree(), getGamma(), getCoef0());

		List<String> ids = new AbstractList<String>(){

			@Override
			public int size(){
				return numberOfVectors;
			}

			@Override
			public String get(int index){
				return String.valueOf(index);
			}
		};

		return LibSVMUtil.createRegression(kernel, new CMatrix<>(xFit, numberOfVectors, numberOfFeatures), ids, null, dualCoef, schema);

	}

	public String getKernel(){
		return getEnum("kernel", this::getString, KernelRidge.ENUM_KERNEL);
	}

	public Integer getDegree(){
		return getInteger("degree");
	}

	public Number getGamma(){
		return getOptionalNumber("gamma");
	}

	public Number getCoef0(){
		return getNumber("coef0");
	}

	public List<Number> getDualCoef(){
		return getNumberArray("dual_coef_");
	}

	public List<Number> getXFit(){
		return getNumberArray("X_fit_");
	}
}