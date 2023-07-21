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
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.neural_network.NeuralEntity;
import org.dmg.pmml.neural_network.NeuralInputs;
import org.dmg.pmml.neural_network.NeuralLayer;
import org.dmg.pmml.neural_network.NeuralNetwork;
import org.dmg.pmml.neural_network.NeuralOutputs;
import org.dmg.pmml.neural_network.Neuron;
import org.jpmml.converter.CMatrixUtil;
import org.jpmml.converter.CategoricalLabel;
import org.jpmml.converter.Feature;
import org.jpmml.converter.Label;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.Schema;
import org.jpmml.converter.neural_network.NeuralNetworkUtil;
import org.jpmml.python.ClassDictUtil;
import org.jpmml.python.HasArray;

public class MultilayerPerceptronUtil {

	private MultilayerPerceptronUtil(){
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

		Label label = schema.getLabel();
		List<? extends Feature> features = schema.getFeatures();

		NeuralInputs neuralInputs = NeuralNetworkUtil.createNeuralInputs(features, DataType.DOUBLE);

		List<NeuralLayer> neuralLayers = encodeNeuralLayers(neuralInputs, coefs, intercepts);

		NeuralOutputs neuralOutputs = encodeNeuralOutputs(miningFunction, neuralLayers, label);

		NeuralNetwork neuralNetwork = new NeuralNetwork(miningFunction, activationFunction, ModelUtil.createMiningSchema(label), neuralInputs, neuralLayers)
			.setNeuralOutputs(neuralOutputs);

		return neuralNetwork;
	}

	static
	public List<NeuralLayer> encodeNeuralLayers(NeuralInputs neuralInputs, List<? extends HasArray> coefs, List<? extends HasArray> intercepts){
		return encodeNeuralLayers(neuralInputs, coefs.size(), coefs, intercepts);
	}

	static
	public List<NeuralLayer> encodeNeuralLayers(NeuralInputs neuralInputs, int numberOfLayers, List<? extends HasArray> coefs, List<? extends HasArray> intercepts){
		ClassDictUtil.checkSize(coefs, intercepts);

		List<? extends NeuralEntity> entities = neuralInputs.getNeuralInputs();

		List<NeuralLayer> result = new ArrayList<>();

		for(int layer = 0; layer < numberOfLayers; layer++){
			HasArray coef = coefs.get(layer);
			HasArray intercept = intercepts.get(layer);

			int[] shape = coef.getArrayShape();

			int rows = shape[0];
			int columns = shape[1];

			NeuralLayer neuralLayer = new NeuralLayer();

			List<?> coefMatrix = coef.getArrayContent();
			List<?> interceptVector = intercept.getArrayContent();

			for(int column = 0; column < columns; column++){
				List<? extends Number> weights = (List)CMatrixUtil.getColumn(coefMatrix, rows, columns, column);
				Number bias = (Number)interceptVector.get(column);

				Neuron neuron = NeuralNetworkUtil.createNeuron(entities, weights, bias)
					.setId(String.valueOf(layer + 1) + "/" + String.valueOf(column + 1));

				neuralLayer.addNeurons(neuron);
			}

			result.add(neuralLayer);

			entities = neuralLayer.getNeurons();
		}

		return result;
	}

	static
	public NeuralOutputs encodeNeuralOutputs(MiningFunction miningFunction, List<NeuralLayer> neuralLayers, Label label){
		NeuralLayer neuralLayer = Iterables.getLast(neuralLayers);

		neuralLayer.setActivationFunction(NeuralNetwork.ActivationFunction.IDENTITY);

		List<? extends NeuralEntity> entities = neuralLayer.getNeurons();

		switch(miningFunction){
			case CLASSIFICATION:
				{
					CategoricalLabel categoricalLabel = (CategoricalLabel)label;

					// Binary classification
					if(categoricalLabel.size() == 2){
						List<NeuralLayer> transformationNeuralLayers = NeuralNetworkUtil.createBinaryLogisticTransformation(Iterables.getOnlyElement(entities));

						neuralLayers.addAll(transformationNeuralLayers);

						neuralLayer = Iterables.getLast(transformationNeuralLayers);

						entities = neuralLayer.getNeurons();
					} else

					// Multi-class classification
					if(categoricalLabel.size() > 2){
						neuralLayer.setNormalizationMethod(NeuralNetwork.NormalizationMethod.SOFTMAX);
					} else

					{
						throw new IllegalArgumentException();
					}

					return NeuralNetworkUtil.createClassificationNeuralOutputs(entities, categoricalLabel);
				}
			case REGRESSION:
				{
					return NeuralNetworkUtil.createRegressionNeuralOutputs(entities, label);
				}
			default:
				throw new IllegalArgumentException();
		}
	}

	static
	public NeuralNetwork.ActivationFunction parseActivationFunction(String activation){

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