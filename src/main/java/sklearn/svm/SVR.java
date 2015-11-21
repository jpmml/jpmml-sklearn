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
import numpy.core.NDArrayUtil;
import org.dmg.pmml.Array;
import org.dmg.pmml.Coefficient;
import org.dmg.pmml.Coefficients;
import org.dmg.pmml.FieldRef;
import org.dmg.pmml.Kernel;
import org.dmg.pmml.LinearKernel;
import org.dmg.pmml.MiningFunctionType;
import org.dmg.pmml.MiningSchema;
import org.dmg.pmml.PolynomialKernel;
import org.dmg.pmml.RadialBasisKernel;
import org.dmg.pmml.SigmoidKernel;
import org.dmg.pmml.SupportVector;
import org.dmg.pmml.SupportVectorMachine;
import org.dmg.pmml.SupportVectorMachineModel;
import org.dmg.pmml.SupportVectors;
import org.dmg.pmml.VectorDictionary;
import org.dmg.pmml.VectorFields;
import org.dmg.pmml.VectorInstance;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.sklearn.ClassDictUtil;
import org.jpmml.sklearn.Schema;
import sklearn.Regressor;
import sklearn.ValueUtil;

public class SVR extends Regressor {

	public SVR(String module, String name){
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
		List<? extends Number> supportVectorArrays = getSupportVectors();
		List<? extends Number> dualCoef = getDualCoef();
		List<? extends Number> intercept = getIntercept();

		Kernel kernel = encodeKernel();

		VectorFields vectorFields = new VectorFields();

		for(int i = 0; i < numberOfFeatures; i++){
			FieldRef fieldRef = new FieldRef(schema.getActiveField(i));

			vectorFields.addFieldRefs(fieldRef);
		}

		VectorDictionary vectorDictionary = new VectorDictionary(vectorFields);

		Coefficients coefficients = new Coefficients()
			.setAbsoluteValue(ValueUtil.asDouble(Iterables.getOnlyElement(intercept)));

		SupportVectors supportVectors = new SupportVectors();

		for(int i = 0; i < numberOfVectors; i++){
			String id = String.valueOf(support.get(i));

			Array array = ValueUtil.encodeArray(NDArrayUtil.getRow(supportVectorArrays, numberOfVectors, numberOfFeatures, i));

			VectorInstance vectorInstance = new VectorInstance(id)
				.setArray(array);

			vectorDictionary.addVectorInstances(vectorInstance);

			Coefficient coefficient = new Coefficient()
				.setValue(ValueUtil.asDouble(dualCoef.get(i)));

			coefficients.addCoefficients(coefficient);

			SupportVector supportVector = new SupportVector(id);

			supportVectors.addSupportVectors(supportVector);
		}

		SupportVectorMachine supportVectorMachine = new SupportVectorMachine(coefficients)
			.setSupportVectors(supportVectors);

		MiningSchema miningSchema = PMMLUtil.createMiningSchema(schema.getTargetField(), schema.getActiveFields());

		SupportVectorMachineModel supportVectorMachineModel = new SupportVectorMachineModel(MiningFunctionType.REGRESSION, miningSchema, vectorDictionary, null)
			.setKernel(kernel)
			.addSupportVectorMachines(supportVectorMachine);

		return supportVectorMachineModel;
	}

	private Kernel encodeKernel(){
		String kernel = getKernel();

		Integer degree = getDegree();
		Double gamma = getGamma();
		Double coef0 = getCoef0();

		if(("linear").equals(kernel)){
			return new LinearKernel();
		} else

		if(("poly").equals(kernel)){
			return new PolynomialKernel()
				.setDegree((double)degree)
				.setCoef0(coef0)
				.setGamma(gamma);
		} else

		if(("rbf").equals(kernel)){
			return new RadialBasisKernel()
				.setGamma(gamma);
		} else

		if(("sigmoid").equals(kernel)){
			return new SigmoidKernel()
				.setCoef0(coef0)
				.setGamma(gamma);
		}

		throw new IllegalArgumentException();
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

	public List<? extends Number> getDualCoef(){
		return (List)ClassDictUtil.getArray(this, "dual_coef_");
	}

	public List<? extends Number> getIntercept(){
		return (List)ClassDictUtil.getArray(this, "intercept_");
	}

	private int[] getSupportVectorsShape(){
		return ClassDictUtil.getShape(this, "support_vectors_", 2);
	}
}