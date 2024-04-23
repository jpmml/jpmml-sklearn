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
package sklearn.neighbors;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import com.google.common.base.Function;
import com.google.common.collect.Lists;
import org.dmg.pmml.CityBlock;
import org.dmg.pmml.CompareFunction;
import org.dmg.pmml.ComparisonMeasure;
import org.dmg.pmml.DataType;
import org.dmg.pmml.Euclidean;
import org.dmg.pmml.Measure;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.Minkowski;
import org.dmg.pmml.Output;
import org.dmg.pmml.OutputField;
import org.dmg.pmml.nearest_neighbor.InstanceField;
import org.dmg.pmml.nearest_neighbor.InstanceFields;
import org.dmg.pmml.nearest_neighbor.KNNInput;
import org.dmg.pmml.nearest_neighbor.KNNInputs;
import org.dmg.pmml.nearest_neighbor.NearestNeighborModel;
import org.dmg.pmml.nearest_neighbor.TrainingInstances;
import org.jpmml.converter.CMatrixUtil;
import org.jpmml.converter.CategoricalLabel;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.ContinuousLabel;
import org.jpmml.converter.Feature;
import org.jpmml.converter.Label;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.converter.ScalarLabel;
import org.jpmml.converter.ScalarLabelUtil;
import org.jpmml.converter.Schema;
import org.jpmml.converter.ValueUtil;
import org.jpmml.python.ClassDictUtil;
import sklearn.Estimator;

public class KNeighborsUtil {

	private KNeighborsUtil(){
	}

	static
	public <E extends Estimator & HasTrainingData> int getNumberOfOutputs(E estimator){
		int[] shape = estimator.getYShape();

		// (n_samples)
		if(shape.length == 1){
			return 1;
		} else

		// (n_samples, n_outputs)
		if(shape.length == 2){
			return shape[1];
		} else

		{
			throw new IllegalArgumentException();
		}
	}

	static
	public boolean parseWeights(String weights){

		switch(weights){
			case KNeighborsConstants.WEIGHTS_DISTANCE:
				return true;
			case KNeighborsConstants.WEIGHTS_UNIFORM:
				return false;
			default:
				throw new IllegalArgumentException(weights);
		}
	}

	static
	public <E extends Estimator & HasMetric & HasNumberOfNeighbors & HasTrainingData> NearestNeighborModel encodeNeighbors(E estimator, MiningFunction miningFunction, int numberOfInstances, int numberOfFeatures, Schema schema){
		int numberOfNeighbors = estimator.getNumberOfNeighbors();
		int numberOfOutputs = estimator.getNumberOfOutputs();

		List<? extends Number> fitX = estimator.getFitX();
		List<?> id = estimator.getId();
		List<? extends Number> y = estimator.getY();

		if(id != null){
			ClassDictUtil.checkSize(numberOfInstances, id);
		} // End if

		if(y != null){
			ClassDictUtil.checkSize(numberOfInstances * numberOfOutputs, y);
		}

		Label label = schema.getLabel();
		List<? extends Feature> features = schema.getFeatures();

		Map<String, List<?>> data = new LinkedHashMap<>();

		InstanceFields instanceFields = new InstanceFields();

		if(id != null){
			InstanceField instanceField = new InstanceField(KNeighborsUtil.VARIABLE_ID)
				.setColumn("data:id");

			instanceFields.addInstanceFields(instanceField);

			data.put(instanceField.getColumn(), id);
		}

		List<ScalarLabel> scalarLabels = (label != null ? ScalarLabelUtil.toScalarLabels(label) : Collections.emptyList());

		ClassDictUtil.checkSize(numberOfOutputs, scalarLabels);

		if(numberOfOutputs == 0){
			// Ignored
		} else

		if(numberOfOutputs == 1){
			ScalarLabel scalarLabel = scalarLabels.get(0);

			if(scalarLabel != null){
				InstanceField instanceField = new InstanceField(scalarLabel.getName())
					.setColumn("data:y");

				instanceFields.addInstanceFields(instanceField);

				data.put(instanceField.getColumn(), translateValues(scalarLabel, y));
			}
		} else

		if(numberOfOutputs >= 2){

			for(int i = 0; i < scalarLabels.size(); i++){
				ScalarLabel scalarLabel = scalarLabels.get(i);

				if(scalarLabel != null){
					InstanceField instanceField = new InstanceField(scalarLabel.getName())
						.setColumn("data:y" + String.valueOf(i + 1));

					instanceFields.addInstanceFields(instanceField);

					data.put(instanceField.getColumn(), translateValues(scalarLabel, CMatrixUtil.getColumn(y, numberOfInstances, numberOfOutputs, i)));
				}
			}
		} else

		{
			throw new IllegalArgumentException();
		}

		DataType dataType = estimator.getDataType();

		KNNInputs knnInputs = new KNNInputs();

		for(int i = 0; i < features.size(); i++){
			Feature feature = features.get(i);

			ContinuousFeature continuousFeature = feature.toContinuousFeature(dataType);

			String name = continuousFeature.getName();

			InstanceField instanceField = new InstanceField(name)
				.setColumn("data:x" + String.valueOf(i + 1));

			instanceFields.addInstanceFields(instanceField);

			KNNInput knnInput = new KNNInput(name);

			knnInputs.addKNNInputs(knnInput);

			data.put(instanceField.getColumn(), CMatrixUtil.getColumn(fitX, numberOfInstances, numberOfFeatures, i));
		}

		TrainingInstances trainingInstances = new TrainingInstances(instanceFields, PMMLUtil.createInlineTable(data))
			.setTransformed(true);

		ComparisonMeasure comparisonMeasure = encodeComparisonMeasure(estimator);

		Output output;

		if(numberOfOutputs == 0 || numberOfOutputs == 1){
			output = createOutput(numberOfNeighbors, null);
		} else

		// XXX
		if(numberOfOutputs >= 2){
			output = createOutput(numberOfNeighbors, scalarLabels.get(0));
		} else

		{
			throw new IllegalArgumentException();
		}

		NearestNeighborModel nearestNeighborModel = new NearestNeighborModel(miningFunction, numberOfNeighbors, ModelUtil.createMiningSchema(schema.getLabel()), trainingInstances, comparisonMeasure, knnInputs)
			.setOutput(output);

		if(id != null){
			nearestNeighborModel.setInstanceIdVariable(VARIABLE_ID);
		}

		return nearestNeighborModel;
	}

	static
	private <E extends Estimator & HasMetric> ComparisonMeasure encodeComparisonMeasure(E estimator){
		Measure measure = encodeMeasure(estimator);

		ComparisonMeasure comparisonMeasure = new ComparisonMeasure(ComparisonMeasure.Kind.DISTANCE, measure)
			.setCompareFunction(CompareFunction.ABS_DIFF);

		return comparisonMeasure;
	}

	static
	private <E extends Estimator & HasMetric> Measure encodeMeasure(E estimator){
		String metric = estimator.getMetric();

		switch(metric){
			case KNeighborsConstants.METRIC_EUCLIDEAN:
				return new Euclidean();
			case KNeighborsConstants.METRIC_MANHATTAN:
				return new CityBlock();
			case KNeighborsConstants.METRIC_MINKOWSKI:
				{
					Integer p = estimator.getP();

					switch(p){
						case 1:
							return new CityBlock();
						case 2:
							return new Euclidean();
						default:
							return new Minkowski(p);
					}
				}
			default:
				throw new IllegalArgumentException(metric);
		}
	}

	static
	private Output createOutput(int numberOfNeighbors, ScalarLabel scalarLabel){

		if(numberOfNeighbors == 1){
			return null;
		}

		Output output = ModelUtil.createNeighborOutput(numberOfNeighbors);

		if(scalarLabel != null){
			List<OutputField> outputFields = output.getOutputFields();

			for(OutputField outputField : outputFields){
				outputField.setTargetField(scalarLabel.getName());
			}
		}

		return output;
	}

	static
	private List<?> translateValues(ScalarLabel scalarLabel, List<? extends Number> y){

		if(scalarLabel instanceof ContinuousLabel){
			ContinuousLabel continuousLabel = (ContinuousLabel)scalarLabel;

			return y;
		} else

		if(scalarLabel instanceof CategoricalLabel){
			CategoricalLabel categoricalLabel = (CategoricalLabel)scalarLabel;

			Function<Number, ?> function = new Function<Number, Object>(){

				@Override
				public Object apply(Number number){
					int index = ValueUtil.asInt(number);

					return categoricalLabel.getValue(index);
				}
			};

			return Lists.transform(y, function);
		} else

		{
			throw new IllegalArgumentException();
		}
	}

	private static final String VARIABLE_ID = "id";
}
