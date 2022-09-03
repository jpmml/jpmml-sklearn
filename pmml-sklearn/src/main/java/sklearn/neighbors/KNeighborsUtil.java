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
import org.jpmml.converter.MultiLabel;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.converter.ScalarLabel;
import org.jpmml.converter.Schema;
import org.jpmml.converter.ValueUtil;
import org.jpmml.converter.nearest_neighbor.NearestNeighborModelUtil;
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
	public <E extends Estimator & HasNeighbors & HasTrainingData> NearestNeighborModel encodeNeighbors(E estimator, MiningFunction miningFunction, int numberOfInstances, int numberOfFeatures, Schema schema){
		int numberOfNeighbors = estimator.getNumberOfNeighbors();
		int numberOfOutputs = estimator.getNumberOfOutputs();

		List<? extends Number> fitX = estimator.getFitX();
		List<? extends Number> y = estimator.getY();

		ClassDictUtil.checkSize(numberOfInstances * numberOfOutputs, y);

		Label label = schema.getLabel();
		List<? extends Feature> features = schema.getFeatures();

		Map<String, List<?>> data = new LinkedHashMap<>();

		InstanceFields instanceFields = new InstanceFields();

		if(numberOfOutputs == 1){
			ScalarLabel scalarLabel = (ScalarLabel)label;

			if(scalarLabel != null){
				InstanceField instanceField = new InstanceField(scalarLabel.getName())
					.setColumn("data:y");

				instanceFields.addInstanceFields(instanceField);

				data.put(instanceField.getColumn(), translateValues(scalarLabel, y));
			}
		} else

		if(numberOfOutputs >= 2){
			MultiLabel multiLabel = (MultiLabel)label;

			List<? extends Label> labels = multiLabel.getLabels();

			for(int i = 0; i < labels.size(); i++){
				ScalarLabel scalarLabel = (ScalarLabel)labels.get(i);

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

		NearestNeighborModel nearestNeighborModel = new NearestNeighborModel(miningFunction, numberOfNeighbors, ModelUtil.createMiningSchema(schema.getLabel()), trainingInstances, comparisonMeasure, knnInputs);

		if(numberOfOutputs == 1){
			nearestNeighborModel.setOutput(NearestNeighborModelUtil.createOutput(numberOfNeighbors));
		}

		return nearestNeighborModel;
	}

	static
	private <E extends Estimator & HasNeighbors> ComparisonMeasure encodeComparisonMeasure(E estimator){
		String weights = estimator.getWeights();

		if(!("uniform").equals(weights)){
			throw new IllegalArgumentException(weights);
		}

		Measure measure = encodeMeasure(estimator);

		ComparisonMeasure comparisonMeasure = new ComparisonMeasure(ComparisonMeasure.Kind.DISTANCE, measure)
			.setCompareFunction(CompareFunction.ABS_DIFF);

		return comparisonMeasure;
	}

	static
	private <E extends Estimator & HasNeighbors> Measure encodeMeasure(E estimator){
		String metric = estimator.getMetric();
		int p = estimator.getP();

		switch(metric){
			case "euclidean":
				return new Euclidean();
			case "manhattan":
				return new CityBlock();
			case "minkowski":
				switch(p){
					case 1:
						return new CityBlock();
					case 2:
						return new Euclidean();
					default:
						return new Minkowski(p);
				}
			default:
				throw new IllegalArgumentException(metric);
		}
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
}
