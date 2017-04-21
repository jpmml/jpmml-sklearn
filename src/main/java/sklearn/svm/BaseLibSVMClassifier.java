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

import org.dmg.pmml.support_vector_machine.SupportVectorMachine;
import org.dmg.pmml.support_vector_machine.SupportVectorMachineModel;
import org.jpmml.converter.CMatrix;
import org.jpmml.converter.Schema;
import org.jpmml.converter.ValueUtil;
import org.jpmml.converter.support_vector_machine.LibSVMUtil;
import org.jpmml.sklearn.ClassDictUtil;
import sklearn.Classifier;

abstract
public class BaseLibSVMClassifier extends Classifier {

	public BaseLibSVMClassifier(String module, String name){
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

		SupportVectorMachineModel supportVectorMachineModel = LibSVMUtil.createClassification(new CMatrix<>(ValueUtil.asDoubles(supportVectors), numberOfVectors, numberOfFeatures), supportSizes, SupportVectorMachineUtil.formatIds(support), ValueUtil.asDoubles(intercept), ValueUtil.asDoubles(dualCoef), schema)
			.setKernel(SupportVectorMachineUtil.createKernel(getKernel(), getDegree(), getGamma(), getCoef0()));

		List<SupportVectorMachine> supportVectorMachines = supportVectorMachineModel.getSupportVectorMachines();
		for(SupportVectorMachine supportVectorMachine : supportVectorMachines){
			String category = supportVectorMachine.getTargetCategory();

			// LibSVM: (decisionFunction > 0 ? first : second)
			// PMML: (decisionFunction < 0 ? first : second)
			supportVectorMachine.setTargetCategory(supportVectorMachine.getAlternateTargetCategory());
			supportVectorMachine.setAlternateTargetCategory(category);
		}

		return supportVectorMachineModel;
	}

	public String getKernel(){
		return (String)get("kernel");
	}

	public Integer getDegree(){
		return ValueUtil.asInteger((Number)get("degree"));
	}

	public Double getGamma(){
		return ValueUtil.asDouble((Number)get("_gamma"));
	}

	public Double getCoef0(){
		return ValueUtil.asDouble((Number)get("coef0"));
	}

	public List<Integer> getSupport(){
		return ValueUtil.asIntegers((List)ClassDictUtil.getArray(this, "support_"));
	}

	public List<? extends Number> getSupportVectors(){
		return (List)ClassDictUtil.getArray(this, "support_vectors_");
	}

	public List<Integer> getSupportSizes(){
		return ValueUtil.asIntegers((List)ClassDictUtil.getArray(this, "n_support_"));
	}

	public List<? extends Number> getDualCoef(){
		return (List)ClassDictUtil.getArray(this, "_dual_coef_");
	}

	public List<? extends Number> getIntercept(){
		return (List)ClassDictUtil.getArray(this, "_intercept_");
	}

	private int[] getSupportVectorsShape(){
		return ClassDictUtil.getShape(this, "support_vectors_", 2);
	}

	static
	private <E> List<E> slice(List<E> list, int[] offsets, int index){
		return list.subList(offsets[index], offsets[index + 1]);
	}
}