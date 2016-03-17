/*
 * Copyright (c) 2016 Villu Ruusmann
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
package sklearn.neural_network;

import java.util.ArrayList;
import java.util.List;

import numpy.core.NDArray;
import numpy.core.NDArrayUtil;
import org.dmg.pmml.ActivationFunctionType;
import org.dmg.pmml.Connection;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.Entity;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.FieldRef;
import org.dmg.pmml.MiningFunctionType;
import org.dmg.pmml.MiningSchema;
import org.dmg.pmml.NeuralInput;
import org.dmg.pmml.NeuralInputs;
import org.dmg.pmml.NeuralLayer;
import org.dmg.pmml.NeuralNetwork;
import org.dmg.pmml.NeuralOutput;
import org.dmg.pmml.NeuralOutputs;
import org.dmg.pmml.Neuron;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.ValueUtil;
import org.jpmml.sklearn.Schema;
import sklearn.Regressor;

public class MLPRegressor extends Regressor {

	public MLPRegressor(String module, String name){
		super(module, name);
	}

	@Override
	public int getNumberOfFeatures(){
		List<?> coefs = getCoefs();

		NDArray input = (NDArray)coefs.get(0);

		int[] shape = NDArrayUtil.getShape(input);

		return shape[0];
	}

	@Override
	public NeuralNetwork encodeModel(Schema schema){
		ActivationFunctionType activation = parseActivationFunction(getActivation());
		ActivationFunctionType outActivation = parseActivationFunction(getOutActivation());

		List<?> coefs = getCoefs();
		List<?> intercepts = getIntercepts();

		int numberOfLayers = getNumberOfLayers();
		int numberOfOutputs = getNumberOfOutputs();

		NeuralInputs neuralInputs = new NeuralInputs();

		List<FieldName> activeFields = schema.getActiveFields();
		for(int column = 0; column < activeFields.size(); column++){
			FieldName activeField = activeFields.get(column);

			DerivedField derivedField = new DerivedField(getOpType(), getDataType())
				.setExpression(new FieldRef(activeField));

			NeuralInput neuralInput = new NeuralInput()
				.setId("0/" + (column + 1))
				.setDerivedField(derivedField);

			neuralInputs.addNeuralInputs(neuralInput);
		}

		List<? extends Entity> inputEntities = neuralInputs.getNeuralInputs();

		List<NeuralLayer> neuralLayers = new ArrayList<>();

		for(int layer = 0; layer < (numberOfLayers - 1); layer++){
			NDArray coef = (NDArray)coefs.get(layer);
			NDArray intercept = (NDArray)intercepts.get(layer);

			List<?> coefMatrix = NDArrayUtil.getContent(coef);
			List<?> interceptVector = NDArrayUtil.getContent(intercept);

			int[] shape = NDArrayUtil.getShape(coef);

			int rows = shape[0];
			int columns = shape[1];

			List<Neuron> neurons = new ArrayList<>();

			for(int column = 0; column < columns; column++){
				Neuron neuron = new Neuron()
					.setId((layer + 1) + "/" + (column + 1));

				Double bias = ValueUtil.asDouble((Number)interceptVector.get(column));
				if(!ValueUtil.isZero(bias)){
					neuron.setBias(bias);
				}

				neurons.add(neuron);
			}

			for(int row = 0; row < rows; row++){
				List<?> weights = NDArrayUtil.getRow(coefMatrix, rows, columns, row);

				connect(inputEntities.get(row), neurons, weights);
			}

			NeuralLayer neuralLayer = new NeuralLayer(neurons);

			if(layer == (numberOfLayers - 2) && !(activation).equals(outActivation)){
				neuralLayer.setActivationFunction(outActivation);
			}

			inputEntities = neuralLayer.getNeurons();

			neuralLayers.add(neuralLayer);
		}

		if((numberOfOutputs != 1) || (numberOfOutputs != inputEntities.size())){
			throw new IllegalArgumentException();
		}

		Neuron outputNeuron = (Neuron)inputEntities.get(0);

		FieldName targetField = schema.getTargetField();

		DerivedField derivedField = new DerivedField(getOpType(), getDataType())
			.setExpression(new FieldRef(targetField));

		NeuralOutput neuralOutput = new NeuralOutput()
			.setOutputNeuron(outputNeuron.getId())
			.setDerivedField(derivedField);

		NeuralOutputs neuralOutputs = new NeuralOutputs()
			.addNeuralOutputs(neuralOutput);

		MiningSchema miningSchema = ModelUtil.createMiningSchema(targetField, activeFields);

		NeuralNetwork neuralNetwork = new NeuralNetwork(MiningFunctionType.REGRESSION, activation, miningSchema, neuralInputs, neuralLayers)
			.setNeuralOutputs(neuralOutputs);

		return neuralNetwork;
	}

	public String getActivation(){
		return (String)get("activation");
	}

	public List<?> getCoefs(){
		return (List<?>)get("coefs_");
	}

	public List<?> getIntercepts(){
		return (List<?>)get("intercepts_");
	}

	public int getNumberOfLayers(){
		return ValueUtil.asInt((Number)get("n_layers_"));
	}

	public int getNumberOfOutputs(){
		return ValueUtil.asInt((Number)get("n_outputs_"));
	}

	public String getOutActivation(){
		return (String)get("out_activation_");
	}

	static
	private ActivationFunctionType parseActivationFunction(String activation){

		if(("identity").equals(activation)){
			return ActivationFunctionType.IDENTITY;
		} else

		if(("tanh").equals(activation)){
			return ActivationFunctionType.TANH;
		} else

		if(("logistic").equals(activation)){
			return ActivationFunctionType.LOGISTIC;
		}

		throw new IllegalArgumentException(activation);
	}

	static
	private void connect(Entity input, List<Neuron> neurons, List<?> weights){

		if(neurons.size() != weights.size()){
			throw new IllegalArgumentException();
		}

		for(int i = 0; i < neurons.size(); i++){
			Neuron neuron = neurons.get(i);
			Double weight = ValueUtil.asDouble((Number)weights.get(i));

			Connection connection = new Connection()
				.setFrom(input.getId())
				.setWeight(weight);

			neuron.addConnections(connection);
		}
	}
}