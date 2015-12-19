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
import java.util.BitSet;
import java.util.List;

import numpy.core.NDArrayUtil;
import org.dmg.pmml.Array;
import org.dmg.pmml.Coefficient;
import org.dmg.pmml.Coefficients;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.FieldRef;
import org.dmg.pmml.Kernel;
import org.dmg.pmml.LinearKernel;
import org.dmg.pmml.PolynomialKernel;
import org.dmg.pmml.RadialBasisKernel;
import org.dmg.pmml.RealSparseArray;
import org.dmg.pmml.SigmoidKernel;
import org.dmg.pmml.SupportVector;
import org.dmg.pmml.SupportVectorMachine;
import org.dmg.pmml.SupportVectors;
import org.dmg.pmml.VectorDictionary;
import org.dmg.pmml.VectorFields;
import org.dmg.pmml.VectorInstance;
import org.jpmml.sklearn.LoggerUtil;
import org.jpmml.sklearn.Schema;
import org.jpmml.sklearn.ValueUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class SupportVectorMachineUtil {

	private SupportVectorMachineUtil(){
	}

	static
	public VectorDictionary encodeVectorDictionary(List<Integer> support, List<? extends Number> supportVectors, int numberOfVectors, int numberOfFeatures, Schema schema){
		BitSet features = new BitSet(numberOfFeatures);

		Double defaultValue = Double.valueOf(0d);

		for(int i = 0; i < numberOfVectors; i++){
			List<? extends Number> values = NDArrayUtil.getRow(supportVectors, numberOfVectors, numberOfFeatures, i);

			BitSet vectorFeatures = ValueUtil.getIndices(values, defaultValue);

			// Set bits that correspond to non-default values
			vectorFeatures.flip(0, numberOfFeatures);

			features.or(vectorFeatures);
		}

		int numberOfUsedFeatures = features.cardinality();

		List<FieldName> unusedActiveFields = new ArrayList<>();

		VectorFields vectorFields = new VectorFields();

		for(int i = 0; i < numberOfFeatures; i++){
			FieldName activeField = schema.getActiveField(i);

			if(!features.get(i)){
				unusedActiveFields.add(activeField);

				continue;
			}

			FieldRef fieldRef = new FieldRef(activeField);

			vectorFields.addFieldRefs(fieldRef);
		}

		VectorDictionary vectorDictionary = new VectorDictionary(vectorFields);

		for(int i = 0; i < numberOfVectors; i++){
			String id = String.valueOf(support.get(i));

			VectorInstance vectorInstance = new VectorInstance(id);

			List<? extends Number> values = NDArrayUtil.getRow(supportVectors, numberOfVectors, numberOfFeatures, i);

			if(numberOfUsedFeatures < numberOfFeatures){
				values = ValueUtil.filterByIndices(values, features);
			} // End if

			if(ValueUtil.isSparseArray(values, defaultValue, 0.75d)){
				RealSparseArray sparseArray = ValueUtil.encodeSparseArray(values, defaultValue);

				vectorInstance.setREALSparseArray(sparseArray);
			} else

			{
				Array array = ValueUtil.encodeArray(values);

				vectorInstance.setArray(array);
			}

			vectorDictionary.addVectorInstances(vectorInstance);
		}

		if(!unusedActiveFields.isEmpty()){
			logger.info("Skipped {} active field(s): {}", unusedActiveFields.size(), LoggerUtil.formatNameList(unusedActiveFields));
		}

		return vectorDictionary;
	}

	static
	public SupportVectorMachine encodeSupportVectorMachine(List<VectorInstance> vectorInstances, List<? extends Number> dualCoef, Number intercept){

		if(vectorInstances.size() != dualCoef.size()){
			throw new IllegalArgumentException();
		}

		Coefficients coefficients = new Coefficients()
			.setAbsoluteValue(ValueUtil.asDouble(intercept));

		SupportVectors supportVectors = new SupportVectors();

		for(int i = 0; i < vectorInstances.size(); i++){
			VectorInstance vectorInstance = vectorInstances.get(i);

			Coefficient coefficient = new Coefficient()
				.setValue(ValueUtil.asDouble(dualCoef.get(i)));

			coefficients.addCoefficients(coefficient);

			SupportVector supportVector = new SupportVector(vectorInstance.getId());

			supportVectors.addSupportVectors(supportVector);
		}

		SupportVectorMachine supportVectorMachine = new SupportVectorMachine(coefficients)
			.setSupportVectors(supportVectors);

		return supportVectorMachine;
	}

	static
	public Kernel encodeKernel(String kernel, Integer degree, Double gamma, Double coef0){

		if(("linear").equals(kernel)){
			return new LinearKernel();
		} else

		if(("poly").equals(kernel)){
			return new PolynomialKernel()
				.setDegree(ValueUtil.asDouble(degree))
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

	private static final Logger logger = LoggerFactory.getLogger(SupportVectorMachineUtil.class);
}