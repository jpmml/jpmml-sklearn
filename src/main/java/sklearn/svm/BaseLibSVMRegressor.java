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
import org.dmg.pmml.Kernel;
import org.dmg.pmml.MiningFunctionType;
import org.dmg.pmml.MiningSchema;
import org.dmg.pmml.SupportVectorMachine;
import org.dmg.pmml.SupportVectorMachineModel;
import org.dmg.pmml.VectorDictionary;
import org.dmg.pmml.VectorInstance;
import org.jpmml.converter.FieldCollector;
import org.jpmml.sklearn.ClassDictUtil;
import org.jpmml.sklearn.Schema;
import org.jpmml.sklearn.ValueUtil;
import sklearn.EstimatorUtil;
import sklearn.Regressor;

abstract
public class BaseLibSVMRegressor extends Regressor {

	public BaseLibSVMRegressor(String module, String name){
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
		List<? extends Number> supportVectors = getSupportVectors();
		List<? extends Number> dualCoef = getDualCoef();
		List<? extends Number> intercept = getIntercept();

		VectorDictionary vectorDictionary = SupportVectorMachineUtil.encodeVectorDictionary(support, supportVectors, numberOfVectors, numberOfFeatures, schema);

		List<VectorInstance> vectorInstances = vectorDictionary.getVectorInstances();

		Kernel kernel = SupportVectorMachineUtil.encodeKernel(getKernel(), getDegree(), getGamma(), getCoef0());

		List<SupportVectorMachine> supportVectorMachines = new ArrayList<>();

		SupportVectorMachine supportVectorMachine = SupportVectorMachineUtil.encodeSupportVectorMachine(vectorInstances, dualCoef, Iterables.getOnlyElement(intercept));

		supportVectorMachines.add(supportVectorMachine);

		FieldCollector fieldCollector = new SupportVectorMachineModelFieldCollector();
		fieldCollector.applyTo(vectorDictionary);

		MiningSchema miningSchema = EstimatorUtil.encodeMiningSchema(schema, fieldCollector);

		SupportVectorMachineModel supportVectorMachineModel = new SupportVectorMachineModel(MiningFunctionType.REGRESSION, miningSchema, vectorDictionary, supportVectorMachines)
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

	public List<? extends Number> getDualCoef(){
		return (List)ClassDictUtil.getArray(this, "_dual_coef_");
	}

	public List<? extends Number> getIntercept(){
		return (List)ClassDictUtil.getArray(this, "_intercept_");
	}

	private int[] getSupportVectorsShape(){
		return ClassDictUtil.getShape(this, "support_vectors_", 2);
	}
}