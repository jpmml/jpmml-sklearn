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

import com.google.common.collect.Iterables;
import numpy.core.NDArray;
import numpy.core.NDArrayUtil;
import org.dmg.pmml.ActivationFunctionType;
import org.dmg.pmml.Connection;
import org.dmg.pmml.DataType;
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
import org.dmg.pmml.NnNormalizationMethodType;
import org.dmg.pmml.NormDiscrete;
import org.dmg.pmml.OpType;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.ValueUtil;
import org.jpmml.sklearn.Schema;

public class NeuralNetworkUtil {

	private NeuralNetworkUtil(){
	}

	static
	public NeuralNetwork encodeNeuralNetwork(MiningFunctionType miningFunction, String activation, List<?> coefs, List<?> intercepts, Schema schema){
		ActivationFunctionType activationFunction = parseActivationFunction(activation);

		if(coefs.size() != intercepts.size()){
			throw new IllegalArgumentException();
		}

		NeuralInputs neuralInputs = new NeuralInputs();

		List<FieldName> activeFields = schema.getActiveFields();
		for(int column = 0; column < activeFields.size(); column++){
			FieldName activeField = activeFields.get(column);

			DerivedField derivedField = new DerivedField(OpType.CONTINUOUS, DataType.DOUBLE)
				.setExpression(new FieldRef(activeField));

			NeuralInput neuralInput = new NeuralInput()
				.setId("0/" + (column + 1))
				.setDerivedField(derivedField);

			neuralInputs.addNeuralInputs(neuralInput);
		}

		List<? extends Entity> entities = neuralInputs.getNeuralInputs();

		List<NeuralLayer> neuralLayers = new ArrayList<>();

		for(int layer = 0; layer < coefs.size(); layer++){
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

				connect(entities.get(row), neurons, weights);
			}

			NeuralLayer neuralLayer = new NeuralLayer(neurons);

			if(layer == (coefs.size() - 1)){
				neuralLayer.setActivationFunction(ActivationFunctionType.IDENTITY);

				switch(miningFunction){
					case REGRESSION:
						break;
					case CLASSIFICATION:
						List<String> targetCategories = schema.getTargetCategories();

						// Binary classification
						if(targetCategories.size() == 2){
							neuralLayers.add(neuralLayer);

							neuralLayer = encodeLogisticTransform(getOnlyNeuron(neuralLayer));

							neuralLayers.add(neuralLayer);

							neuralLayer = encodeLabelBinarizerTransform(getOnlyNeuron(neuralLayer));
						} else

						// Multi-class classification
						{
							neuralLayer.setNormalizationMethod(NnNormalizationMethodType.SOFTMAX);
						}
						break;
					default:
						break;
				}
			}

			entities = neuralLayer.getNeurons();

			neuralLayers.add(neuralLayer);
		}

		NeuralOutputs neuralOutputs = null;

		switch(miningFunction){
			case REGRESSION:
				neuralOutputs = encodeRegressionNeuralOutputs(entities, schema);
				break;
			case CLASSIFICATION:
				neuralOutputs = encodeClassificationNeuralOutputs(entities, schema);
				break;
			default:
				break;
		}

		MiningSchema miningSchema = ModelUtil.createMiningSchema(schema.getTargetField(), activeFields);

		NeuralNetwork neuralNetwork = new NeuralNetwork(miningFunction, activationFunction, miningSchema, neuralInputs, neuralLayers)
			.setNeuralOutputs(neuralOutputs);

		return neuralNetwork;
	}

	static
	private NeuralLayer encodeLogisticTransform(Neuron input){
		NeuralLayer neuralLayer = new NeuralLayer()
			.setActivationFunction(ActivationFunctionType.LOGISTIC);

		Neuron neuron = new Neuron()
			.setId("logistic/1")
			.setBias(0d)
			.addConnections(new Connection(input.getId(), 1d));

		neuralLayer.addNeurons(neuron);

		return neuralLayer;
	}

	static
	private NeuralLayer encodeLabelBinarizerTransform(Neuron input){
		NeuralLayer neuralLayer = new NeuralLayer()
			.setActivationFunction(ActivationFunctionType.IDENTITY);

		Neuron noEventNeuron = new Neuron()
			.setId("event/false")
			.setBias(1d)
			.addConnections(new Connection(input.getId(), -1d));

		Neuron eventNeuron = new Neuron()
			.setId("event/true")
			.setBias(0d)
			.addConnections(new Connection(input.getId(), 1d));

		neuralLayer.addNeurons(noEventNeuron, eventNeuron);

		return neuralLayer;
	}

	static
	private NeuralOutputs encodeRegressionNeuralOutputs(List<? extends Entity> entities, Schema schema){
		FieldName targetField = schema.getTargetField();

		if(entities.size() != 1){
			throw new IllegalArgumentException();
		}

		Entity entity = Iterables.getOnlyElement(entities);

		DerivedField derivedField = new DerivedField(OpType.CONTINUOUS, DataType.DOUBLE)
			.setExpression(new FieldRef(targetField));

		NeuralOutput neuralOutput = new NeuralOutput()
			.setOutputNeuron(entity.getId())
			.setDerivedField(derivedField);

		NeuralOutputs neuralOutputs = new NeuralOutputs()
			.addNeuralOutputs(neuralOutput);

		return neuralOutputs;
	}

	static
	private NeuralOutputs encodeClassificationNeuralOutputs(List<? extends Entity> entities, Schema schema){
		FieldName targetField = schema.getTargetField();

		List<String> targetCategories = schema.getTargetCategories();
		if(entities.size() != targetCategories.size()){
			throw new IllegalArgumentException();
		}

		NeuralOutputs neuralOutputs = new NeuralOutputs();

		for(int i = 0; i < targetCategories.size(); i++){
			Entity entity = entities.get(i);

			String targetCategory = targetCategories.get(i);

			DerivedField derivedField = new DerivedField(OpType.CATEGORICAL, DataType.STRING)
				.setExpression(new NormDiscrete(targetField, targetCategory));

			NeuralOutput neuralOutput = new NeuralOutput()
				.setOutputNeuron(entity.getId())
				.setDerivedField(derivedField);

			neuralOutputs.addNeuralOutputs(neuralOutput);
		}

		return neuralOutputs;
	}

	static
	private void connect(Entity input, List<Neuron> neurons, List<?> weights){

		if(neurons.size() != weights.size()){
			throw new IllegalArgumentException();
		}

		for(int i = 0; i < neurons.size(); i++){
			Neuron neuron = neurons.get(i);
			Double weight = ValueUtil.asDouble((Number)weights.get(i));

			neuron.addConnections(new Connection(input.getId(), weight));
		}
	}

	static
	private Neuron getOnlyNeuron(NeuralLayer neuralLayer){
		List<Neuron> neurons = neuralLayer.getNeurons();

		return Iterables.getOnlyElement(neurons);
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
}