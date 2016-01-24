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

import java.util.ArrayList;
import java.util.List;

import com.google.common.collect.Iterables;
import numpy.core.NDArrayUtil;
import org.dmg.pmml.Kernel;
import org.dmg.pmml.MiningFunctionType;
import org.dmg.pmml.MiningSchema;
import org.dmg.pmml.SupportVectorMachine;
import org.dmg.pmml.SupportVectorMachineModel;
import org.dmg.pmml.SvmClassificationMethodType;
import org.dmg.pmml.VectorDictionary;
import org.dmg.pmml.VectorInstance;
import org.jpmml.converter.ValueUtil;
import org.jpmml.model.visitors.FieldReferenceFinder;
import org.jpmml.sklearn.ClassDictUtil;
import org.jpmml.sklearn.Schema;
import sklearn.Classifier;
import sklearn.EstimatorUtil;

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

		int[] offsets = new int[supportSizes.size() + 1];

		for(int i = 0; i < supportSizes.size(); i++){
			offsets[i + 1] = offsets[i] + supportSizes.get(i);
		}

		VectorDictionary vectorDictionary = SupportVectorMachineUtil.encodeVectorDictionary(support, supportVectors, numberOfVectors, numberOfFeatures, schema);

		List<VectorInstance> vectorInstances = vectorDictionary.getVectorInstances();

		Kernel kernel = SupportVectorMachineUtil.encodeKernel(getKernel(), getDegree(), getGamma(), getCoef0());

		List<SupportVectorMachine> supportVectorMachines = new ArrayList<>();

		int i = 0;

		List<String> targetCategories = schema.getTargetCategories();

		for(int first = 0; first < targetCategories.size(); first++){

			for(int second = first + 1; second < targetCategories.size(); second++){
				List<VectorInstance> svmVectorInstances = new ArrayList<>();
				svmVectorInstances.addAll(slice(vectorInstances, offsets, first));
				svmVectorInstances.addAll(slice(vectorInstances, offsets, second));

				List<Number> svmDualCoef = new ArrayList<>();
				svmDualCoef.addAll(slice(NDArrayUtil.getRow(dualCoef, targetCategories.size() - 1, numberOfVectors, second - 1), offsets, first));
				svmDualCoef.addAll(slice(NDArrayUtil.getRow(dualCoef, targetCategories.size() - 1, numberOfVectors, first), offsets, second));

				// LibSVM: (decisionFunction > 0 ? first : second)
				// PMML: (decisionFunction < 0 ? first : second)
				SupportVectorMachine supportVectorMachine = SupportVectorMachineUtil.encodeSupportVectorMachine(svmVectorInstances, svmDualCoef, Iterables.get(intercept, i))
					.setTargetCategory(targetCategories.get(second))
					.setAlternateTargetCategory(targetCategories.get(first));

				supportVectorMachines.add(supportVectorMachine);

				i++;
			}
		}

		FieldReferenceFinder fieldReferenceFinder = new FieldReferenceFinder();
		fieldReferenceFinder.applyTo(vectorDictionary);

		MiningSchema miningSchema = EstimatorUtil.encodeMiningSchema(schema, fieldReferenceFinder);

		SupportVectorMachineModel supportVectorMachineModel = new SupportVectorMachineModel(MiningFunctionType.CLASSIFICATION, miningSchema, vectorDictionary, supportVectorMachines)
			.setClassificationMethod(SvmClassificationMethodType.ONE_AGAINST_ONE)
			.setKernel(kernel);

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