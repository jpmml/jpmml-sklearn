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

import org.dmg.pmml.support_vector_machine.Kernel;
import org.dmg.pmml.support_vector_machine.SupportVectorMachine;
import org.dmg.pmml.support_vector_machine.SupportVectorMachineModel;
import org.jpmml.converter.CMatrix;
import org.jpmml.converter.Schema;
import org.jpmml.converter.support_vector_machine.LibSVMUtil;
import sklearn.Classifier;

public class LibSVMClassifier extends Classifier {

	public LibSVMClassifier(String module, String name){
		super(module, name);
	}

	@Override
	public int getNumberOfFeatures(){
		int[] shape = getSupportVectorsShape();

		return shape[1];
	}

	@Override
	public boolean hasProbabilityDistribution(){
		return false;
	}

	@Override
	public SupportVectorMachineModel encodeModel(Schema schema){
		int[] shape = getSupportVectorsShape();

		int numberOfVectors = shape[0];
		int numberOfFeatures = shape[1];

		List<Integer> support = getSupport();
		List<? extends Number> supportVectors = getSupportVectors();
		List<Integer> supportSizes = getSupportSizes();
		List<? extends Number> dualCoef = getDualCoef();
		List<? extends Number> intercept = getIntercept();

		Kernel kernel = SupportVectorMachineUtil.createKernel(getKernel(), getDegree(), getGamma(), getCoef0());

		SupportVectorMachineModel supportVectorMachineModel = LibSVMUtil.createClassification(kernel, new CMatrix<>(supportVectors, numberOfVectors, numberOfFeatures), supportSizes, SupportVectorMachineUtil.formatIds(support), intercept, dualCoef, schema);

		List<SupportVectorMachine> supportVectorMachines = supportVectorMachineModel.getSupportVectorMachines();
		for(SupportVectorMachine supportVectorMachine : supportVectorMachines){
			Object category = supportVectorMachine.getTargetCategory();
			Object alternateTargetCategory = supportVectorMachine.getAlternateTargetCategory();

			// LibSVM: (decisionFunction > 0 ? first : second)
			// PMML: (decisionFunction < 0 ? first : second)
			supportVectorMachine.setTargetCategory(alternateTargetCategory);
			supportVectorMachine.setAlternateTargetCategory(category);
		}

		return supportVectorMachineModel;
	}

	public String getKernel(){
		return getString("kernel");
	}

	public Integer getDegree(){
		return getInteger("degree");
	}

	public Number getGamma(){
		return getNumber("_gamma");
	}

	public Number getCoef0(){
		return getNumber("coef0");
	}

	public List<Integer> getSupport(){
		return getIntegerArray("support_");
	}

	public List<? extends Number> getSupportVectors(){
		return getNumberArray("support_vectors_");
	}

	public int[] getSupportVectorsShape(){
		return getArrayShape("support_vectors_", 2);
	}

	public List<Integer> getSupportSizes(){

		// SkLearn 0.21
		if(containsKey("n_support_")){
			return getIntegerArray("n_support_");
		}

		// SkLearn 0.22+
		return getIntegerArray("_n_support");
	}

	public List<? extends Number> getDualCoef(){
		return getNumberArray("_dual_coef_");
	}

	public List<? extends Number> getIntercept(){
		return getNumberArray("_intercept_");
	}
}