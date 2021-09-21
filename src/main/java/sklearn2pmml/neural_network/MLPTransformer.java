/*
 * Copyright (c) 2021 Villu Ruusmann
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
package sklearn2pmml.neural_network;

import java.util.ArrayList;
import java.util.List;

import com.google.common.collect.Iterables;
import org.dmg.pmml.DataField;
import org.dmg.pmml.DataType;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.FieldRef;
import org.dmg.pmml.MiningField;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.MiningSchema;
import org.dmg.pmml.OpType;
import org.dmg.pmml.OutputField;
import org.dmg.pmml.neural_network.NeuralInputs;
import org.dmg.pmml.neural_network.NeuralLayer;
import org.dmg.pmml.neural_network.NeuralNetwork;
import org.dmg.pmml.neural_network.NeuralOutput;
import org.dmg.pmml.neural_network.NeuralOutputs;
import org.dmg.pmml.neural_network.Neuron;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.DerivedOutputField;
import org.jpmml.converter.Feature;
import org.jpmml.converter.FieldNameUtil;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.neural_network.NeuralNetworkUtil;
import org.jpmml.python.HasArray;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn.Estimator;
import sklearn.Transformer;
import sklearn.neural_network.MLPRegressor;
import sklearn.neural_network.MultilayerPerceptronUtil;

public class MLPTransformer extends Transformer {

	public MLPTransformer(String module, String name){
		super(module, name);
	}

	@Override
	public List<Feature> encodeFeatures(List<Feature> features, SkLearnEncoder encoder){
		MLPRegressor mlp = getMLP();
		int transformerOutputLayer = getTransformerOutputLayer();

		String activation = mlp.getActivation();

		NeuralNetwork.ActivationFunction activationFunction = MultilayerPerceptronUtil.parseActivationFunction(activation);

		List<? extends HasArray> coefs = mlp.getCoefs();
		List<? extends HasArray> intercepts = mlp.getIntercepts();

		MiningSchema miningSchema = new MiningSchema();

		NeuralInputs neuralInputs = NeuralNetworkUtil.createNeuralInputs(features, DataType.DOUBLE);

		List<NeuralLayer> neuralLayers;

		if(transformerOutputLayer < 0){
			neuralLayers = MultilayerPerceptronUtil.encodeNeuralLayers(neuralInputs, coefs, intercepts);
		} else

		{
			neuralLayers = MultilayerPerceptronUtil.encodeNeuralLayers(neuralInputs, transformerOutputLayer, coefs, intercepts);
		}

		NeuralOutputs neuralOutputs = new NeuralOutputs();

		NeuralLayer neuralLayer = Iterables.getLast(neuralLayers);

		neuralLayer.setActivationFunction(NeuralNetwork.ActivationFunction.IDENTITY);

		List<Neuron> neurons = neuralLayer.getNeurons();

		List<DataField> dataFields = new ArrayList<>();

		for(int i = 0; i < neurons.size(); i++){
			Neuron neuron = neurons.get(i);

			DataField dataField = encoder.createDataField(FieldNameUtil.create("mlp", i), OpType.CONTINUOUS, DataType.DOUBLE);

			MiningField miningField = ModelUtil.createMiningField(dataField.getName(), MiningField.UsageType.TARGET);

			miningSchema.addMiningFields(miningField);

			DerivedField derivedField = new DerivedField(null, OpType.CONTINUOUS, DataType.DOUBLE, new FieldRef(dataField.getName()));

			NeuralOutput neuralOutput = new NeuralOutput()
				.setOutputNeuron(neuron.getId())
				.setDerivedField(derivedField);

			neuralOutputs.addNeuralOutputs(neuralOutput);

			dataFields.add(dataField);
		}

		NeuralNetwork neuralNetwork = new NeuralNetwork(MiningFunction.REGRESSION, activationFunction, miningSchema, neuralInputs, neuralLayers)
			.setNeuralOutputs(neuralOutputs);

		encoder.addTransformer(neuralNetwork);

		List<Feature> result = new ArrayList<>();

		for(int i = 0; i < dataFields.size(); i++){
			DataField dataField = dataFields.get(i);

			OutputField outputField = ModelUtil.createPredictedField(FieldNameUtil.create(Estimator.FIELD_PREDICT, dataField.getName()), dataField.getOpType(), dataField.getDataType())
				.setTargetField(dataField.getName())
				.setFinalResult(false);

			DerivedOutputField derivedOutputField = encoder.createDerivedField(neuralNetwork, outputField, false);

			result.add(new ContinuousFeature(encoder, derivedOutputField));
		}

		return result;
	}

	public MLPRegressor getMLP(){
		return get("mlp_", MLPRegressor.class);
	}

	public int getTransformerOutputLayer(){
		return getInteger("transformer_output_layer");
	}
}