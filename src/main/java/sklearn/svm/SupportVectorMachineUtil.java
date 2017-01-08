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

import org.dmg.pmml.Array;
import org.dmg.pmml.RealSparseArray;
import org.dmg.pmml.regression.CategoricalPredictor;
import org.dmg.pmml.support_vector_machine.Coefficient;
import org.dmg.pmml.support_vector_machine.Coefficients;
import org.dmg.pmml.support_vector_machine.Kernel;
import org.dmg.pmml.support_vector_machine.LinearKernel;
import org.dmg.pmml.support_vector_machine.PolynomialKernel;
import org.dmg.pmml.support_vector_machine.RadialBasisKernel;
import org.dmg.pmml.support_vector_machine.SigmoidKernel;
import org.dmg.pmml.support_vector_machine.SupportVector;
import org.dmg.pmml.support_vector_machine.SupportVectorMachine;
import org.dmg.pmml.support_vector_machine.SupportVectors;
import org.dmg.pmml.support_vector_machine.VectorDictionary;
import org.dmg.pmml.support_vector_machine.VectorFields;
import org.dmg.pmml.support_vector_machine.VectorInstance;
import org.jpmml.converter.BinaryFeature;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.Feature;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.converter.Schema;
import org.jpmml.converter.ValueUtil;
import org.jpmml.sklearn.ClassDictUtil;
import org.jpmml.sklearn.LoggerUtil;
import org.jpmml.sklearn.MatrixUtil;
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
			List<? extends Number> values = MatrixUtil.getRow(supportVectors, numberOfVectors, numberOfFeatures, i);

			BitSet vectorFeatures = ValueUtil.getIndices(values, defaultValue);

			// Set bits that correspond to non-default values
			vectorFeatures.flip(0, numberOfFeatures);

			features.or(vectorFeatures);
		}

		int numberOfUsedFeatures = features.cardinality();

		List<Feature> unusedFeatures = new ArrayList<>();

		VectorFields vectorFields = new VectorFields();

		for(int i = 0; i < numberOfFeatures; i++){
			Feature feature = schema.getFeature(i);

			if(!features.get(i)){
				unusedFeatures.add(feature);

				continue;
			} // End if

			if(feature instanceof BinaryFeature){
				BinaryFeature binaryFeature = (BinaryFeature)feature;

				CategoricalPredictor categoricalPredictor = new CategoricalPredictor(binaryFeature.getName(), binaryFeature.getValue(), 1d);

				vectorFields.addContent(categoricalPredictor);
			} else

			{
				ContinuousFeature continuousFeature = feature.toContinuousFeature();

				vectorFields.addContent(continuousFeature.ref());
			}
		}

		VectorDictionary vectorDictionary = new VectorDictionary(vectorFields);

		for(int i = 0; i < numberOfVectors; i++){
			String id = String.valueOf(support.get(i));

			VectorInstance vectorInstance = new VectorInstance(id);

			List<? extends Number> values = MatrixUtil.getRow(supportVectors, numberOfVectors, numberOfFeatures, i);

			if(numberOfUsedFeatures < numberOfFeatures){
				values = ValueUtil.filterByIndices(values, features);
			} // End if

			if(ValueUtil.isSparse(values, defaultValue, 0.75d)){
				RealSparseArray sparseArray = PMMLUtil.createRealSparseArray(values, defaultValue);

				vectorInstance.setRealSparseArray(sparseArray);
			} else

			{
				Array array = PMMLUtil.createRealArray(values);

				vectorInstance.setArray(array);
			}

			vectorDictionary.addVectorInstances(vectorInstance);
		}

		if(!unusedFeatures.isEmpty()){
			logger.info("Skipped {} feature(s): {}", unusedFeatures.size(), LoggerUtil.formatNameList(unusedFeatures));
		}

		return vectorDictionary;
	}

	static
	public SupportVectorMachine encodeSupportVectorMachine(List<VectorInstance> vectorInstances, List<? extends Number> dualCoef, Number intercept){
		ClassDictUtil.checkSize(vectorInstances, dualCoef);

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

		switch(kernel){
			case "linear":
				return new LinearKernel();
			case "poly":
				return new PolynomialKernel()
					.setDegree(ValueUtil.asDouble(degree))
					.setCoef0(coef0)
					.setGamma(gamma);
			case "rbf":
				return new RadialBasisKernel()
					.setGamma(gamma);
			case "sigmoid":
				return new SigmoidKernel()
					.setCoef0(coef0)
					.setGamma(gamma);
			default:
				throw new IllegalArgumentException(kernel);
		}
	}

	private static final Logger logger = LoggerFactory.getLogger(SupportVectorMachineUtil.class);
}