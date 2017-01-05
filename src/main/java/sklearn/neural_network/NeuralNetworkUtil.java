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
import org.dmg.pmml.DataType;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.Entity;
import org.dmg.pmml.FieldRef;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.NormDiscrete;
import org.dmg.pmml.OpType;
import org.dmg.pmml.neural_network.Connection;
import org.dmg.pmml.neural_network.NeuralInput;
import org.dmg.pmml.neural_network.NeuralInputs;
import org.dmg.pmml.neural_network.NeuralLayer;
import org.dmg.pmml.neural_network.NeuralNetwork;
import org.dmg.pmml.neural_network.NeuralOutput;
import org.dmg.pmml.neural_network.NeuralOutputs;
import org.dmg.pmml.neural_network.Neuron;
import org.jpmml.converter.CategoricalLabel;
import org.jpmml.converter.ContinuousLabel;
import org.jpmml.converter.Feature;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.Schema;
import org.jpmml.converter.ValueUtil;
import org.jpmml.sklearn.HasArray;
import org.jpmml.sklearn.MatrixUtil;

public class NeuralNetworkUtil {

	private NeuralNetworkUtil(){
	}

	static
	public int getNumberOfFeatures(List<? extends HasArray> coefs){
		HasArray input = coefs.get(0);

		int[] shape = input.getArrayShape();
		if(shape.length != 2){
			throw new IllegalArgumentException();
		}

		return shape[0];
	}

	static
	public NeuralNetwork encodeNeuralNetwork(MiningFunction miningFunction, String activation, List<? extends HasArray> coefs, List<? extends HasArray> intercepts, Schema schema){
		NeuralNetwork.ActivationFunction activationFunction = parseActivationFunction(activation);

		if(coefs.size() != intercepts.size()){
			throw new IllegalArgumentException();
		}

		NeuralInputs neuralInputs = new NeuralInputs();

		List<Feature> features = schema.getFeatures();
		for(int column = 0; column < features.size(); column++){
			Feature feature = features.get(column);

			DerivedField derivedField = new DerivedField(OpType.CONTINUOUS, DataType.DOUBLE)
				.setExpression(feature.ref());

			NeuralInput neuralInput = new NeuralInput()
				.setId("0/" + (column + 1))
				.setDerivedField(derivedField);

			neuralInputs.addNeuralInputs(neuralInput);
		}

		List<? extends Entity> entities = neuralInputs.getNeuralInputs();

		List<NeuralLayer> neuralLayers = new ArrayList<>();

		for(int layer = 0; layer < coefs.size(); layer++){
			HasArray coef = coefs.get(layer);
			HasArray intercept = intercepts.get(layer);

			int[] shape = coef.getArrayShape();

			int rows = shape[0];
			int columns = shape[1];

			List<Neuron> neurons = new ArrayList<>();

			List<?> interceptVector = intercept.getArrayContent();

			for(int column = 0; column < columns; column++){
				Neuron neuron = new Neuron()
					.setId((layer + 1) + "/" + (column + 1));

				Double bias = ValueUtil.asDouble((Number)interceptVector.get(column));
				if(!ValueUtil.isZero(bias)){
					neuron.setBias(bias);
				}

				neurons.add(neuron);
			}

			List<?> coefMatrix = coef.getArrayContent();

			for(int row = 0; row < rows; row++){
				List<?> weights = MatrixUtil.getRow(coefMatrix, rows, columns, row);

				connect(entities.get(row), neurons, weights);
			}

			NeuralLayer neuralLayer = new NeuralLayer(neurons);

			if(layer == (coefs.size() - 1)){
				neuralLayer.setActivationFunction(NeuralNetwork.ActivationFunction.IDENTITY);

				switch(miningFunction){
					case REGRESSION:
						break;
					case CLASSIFICATION:
						CategoricalLabel categoricalLabel = (CategoricalLabel)schema.getLabel();

						// Binary classification
						if(categoricalLabel.size() == 2){
							neuralLayers.add(neuralLayer);

							neuralLayer = encodeLogisticTransform(getOnlyNeuron(neuralLayer));

							neuralLayers.add(neuralLayer);

							neuralLayer = encodeLabelBinarizerTransform(getOnlyNeuron(neuralLayer));
						} else

						// Multi-class classification
						if(categoricalLabel.size() > 2){
							neuralLayer.setNormalizationMethod(NeuralNetwork.NormalizationMethod.SOFTMAX);
						} else

						{
							throw new IllegalArgumentException();
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

		NeuralNetwork neuralNetwork = new NeuralNetwork(miningFunction, activationFunction, ModelUtil.createMiningSchema(schema), neuralInputs, neuralLayers)
			.setNeuralOutputs(neuralOutputs);

		return neuralNetwork;
	}

	static
	private NeuralLayer encodeLogisticTransform(Neuron input){
		NeuralLayer neuralLayer = new NeuralLayer()
			.setActivationFunction(NeuralNetwork.ActivationFunction.LOGISTIC);

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
			.setActivationFunction(NeuralNetwork.ActivationFunction.IDENTITY);

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
		ContinuousLabel continuousLabel = (ContinuousLabel)schema.getLabel();

		if(entities.size() != 1){
			throw new IllegalArgumentException();
		}

		Entity entity = Iterables.getOnlyElement(entities);

		DerivedField derivedField = new DerivedField(OpType.CONTINUOUS, DataType.DOUBLE)
			.setExpression(new FieldRef(continuousLabel.getName()));

		NeuralOutput neuralOutput = new NeuralOutput()
			.setOutputNeuron(entity.getId())
			.setDerivedField(derivedField);

		NeuralOutputs neuralOutputs = new NeuralOutputs()
			.addNeuralOutputs(neuralOutput);

		return neuralOutputs;
	}

	static
	private NeuralOutputs encodeClassificationNeuralOutputs(List<? extends Entity> entities, Schema schema){
		CategoricalLabel categoricalLabel = (CategoricalLabel)schema.getLabel();

		if(categoricalLabel.size() != entities.size()){
			throw new IllegalArgumentException();
		}

		NeuralOutputs neuralOutputs = new NeuralOutputs();

		for(int i = 0; i < categoricalLabel.size(); i++){
			Entity entity = entities.get(i);

			DerivedField derivedField = new DerivedField(OpType.CATEGORICAL, DataType.STRING)
				.setExpression(new NormDiscrete(categoricalLabel.getName(), categoricalLabel.getValue(i)));

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
	private NeuralNetwork.ActivationFunction parseActivationFunction(String activation){

		switch(activation){
			case "identity":
				return NeuralNetwork.ActivationFunction.IDENTITY;
			case "logistic":
				return NeuralNetwork.ActivationFunction.LOGISTIC;
			case "relu":
				return NeuralNetwork.ActivationFunction.RECTIFIER;
			case "tanh":
				return NeuralNetwork.ActivationFunction.TANH;
			default:
				throw new IllegalArgumentException(activation);
		}
	}
}