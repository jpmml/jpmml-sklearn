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

import com.google.common.collect.Iterables;
import org.dmg.pmml.support_vector_machine.Kernel;
import org.dmg.pmml.support_vector_machine.SupportVectorMachineModel;
import org.jpmml.converter.CMatrix;
import org.jpmml.converter.Schema;
import org.jpmml.converter.support_vector_machine.LibSVMUtil;
import sklearn.SkLearnRegressor;

public class LibSVMRegressor extends SkLearnRegressor implements HasLibSVMKernel {

	public LibSVMRegressor(String module, String name){
		super(module, name);
	}

	@Override
	public int getNumberOfFeatures(){
		int[] shape = getSupportVectorsShape();

		return shape[1];
	}

	@Override
	public SupportVectorMachineModel encodeModel(Schema schema){
		int[] shape = getSupportVectorsShape();

		int numberOfVectors = shape[0];
		int numberOfFeatures = shape[1];

		List<Integer> support = getSupport();
		List<Number> supportVectors = getSupportVectors();
		List<Number> dualCoef = getDualCoef();
		List<Number> intercept = getIntercept();

		Kernel kernel = SupportVectorMachineUtil.createKernel(this);

		return LibSVMUtil.createRegression(kernel, new CMatrix<>(supportVectors, numberOfVectors, numberOfFeatures), SupportVectorMachineUtil.formatIds(support), Iterables.getOnlyElement(intercept), dualCoef, schema);
	}

	@Override
	public String getKernel(){
		return getEnum("kernel", this::getString, LibSVMRegressor.ENUM_KERNEL);
	}

	@Override
	public Integer getDegree(){
		return getInteger("degree");
	}

	@Override
	public Number getGamma(){
		return getNumber("_gamma");
	}

	@Override
	public Number getCoef0(){
		return getNumber("coef0");
	}

	public List<Integer> getSupport(){
		return getIntegerArray("support_");
	}

	public List<Number> getSupportVectors(){
		return getNumberArray("support_vectors_");
	}

	public int[] getSupportVectorsShape(){
		return getArrayShape("support_vectors_", 2);
	}

	public List<Number> getDualCoef(){
		return getNumberArray("_dual_coef_");
	}

	public List<Number> getIntercept(){
		return getNumberArray("_intercept_");
	}
}