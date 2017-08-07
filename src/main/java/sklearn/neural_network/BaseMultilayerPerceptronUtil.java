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
import org.dmg.pmml.Entity;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.neural_network.Connection;
import org.dmg.pmml.neural_network.NeuralInputs;
import org.dmg.pmml.neural_network.NeuralLayer;
import org.dmg.pmml.neural_network.NeuralNetwork;
import org.dmg.pmml.neural_network.NeuralOutputs;
import org.dmg.pmml.neural_network.Neuron;
import org.jpmml.converter.CMatrixUtil;
import org.jpmml.converter.CategoricalLabel;
import org.jpmml.converter.ContinuousLabel;
import org.jpmml.converter.Feature;
import org.jpmml.converter.Label;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.Schema;
import org.jpmml.converter.ValueUtil;
import org.jpmml.converter.neural_network.NeuralNetworkUtil;
import org.jpmml.sklearn.ClassDictUtil;
import org.jpmml.sklearn.HasArray;

public class BaseMultilayerPerceptronUtil {

	private BaseMultilayerPerceptronUtil(){
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

		ClassDictUtil.checkSize(coefs, intercepts);

		List<Feature> features = schema.getFeatures();

		Label label = schema.getLabel();

		NeuralInputs neuralInputs = NeuralNetworkUtil.createNeuralInputs(features, DataType.DOUBLE);

		List<? extends Entity> entities = neuralInputs.getNeuralInputs();

		List<NeuralLayer> neuralLayers = new ArrayList<>();

		for(int layer = 0; layer < coefs.size(); layer++){
			HasArray coef = coefs.get(layer);
			HasArray intercept = intercepts.get(layer);

			int[] shape = coef.getArrayShape();

			int rows = shape[0];
			int columns = shape[1];

			NeuralLayer neuralLayer = new NeuralLayer();

			List<?> coefMatrix = coef.getArrayContent();
			List<?> interceptVector = intercept.getArrayContent();

			for(int column = 0; column < columns; column++){
				List<Double> weights = (List)CMatrixUtil.getColumn(coefMatrix, rows, columns, column);
				Double bias = ValueUtil.asDouble((Number)interceptVector.get(column));

				Neuron neuron = NeuralNetworkUtil.createNeuron(entities, weights, bias)
					.setId(String.valueOf(layer + 1) + "/" + String.valueOf(column + 1));

				neuralLayer.addNeurons(neuron);
			}

			if(layer == (coefs.size() - 1)){
				neuralLayer.setActivationFunction(NeuralNetwork.ActivationFunction.IDENTITY);

				switch(miningFunction){
					case REGRESSION:
						break;
					case CLASSIFICATION:
						CategoricalLabel categoricalLabel = (CategoricalLabel)label;

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
				neuralOutputs = NeuralNetworkUtil.createRegressionNeuralOutputs(entities, (ContinuousLabel)label);
				break;
			case CLASSIFICATION:
				neuralOutputs = NeuralNetworkUtil.createClassificationNeuralOutputs(entities, (CategoricalLabel)label);
				break;
			default:
				break;
		}

		NeuralNetwork neuralNetwork = new NeuralNetwork(miningFunction, activationFunction, ModelUtil.createMiningSchema(schema.getLabel()), neuralInputs, neuralLayers)
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