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

import org.dmg.pmml.CityBlock;
import org.dmg.pmml.CompareFunction;
import org.dmg.pmml.ComparisonMeasure;
import org.dmg.pmml.DataType;
import org.dmg.pmml.Euclidean;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.Measure;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.Minkowski;
import org.dmg.pmml.OpType;
import org.dmg.pmml.Output;
import org.dmg.pmml.OutputField;
import org.dmg.pmml.ResultFeature;
import org.dmg.pmml.nearest_neighbor.InstanceField;
import org.dmg.pmml.nearest_neighbor.InstanceFields;
import org.dmg.pmml.nearest_neighbor.KNNInput;
import org.dmg.pmml.nearest_neighbor.KNNInputs;
import org.dmg.pmml.nearest_neighbor.NearestNeighborModel;
import org.dmg.pmml.nearest_neighbor.TrainingInstances;
import org.jpmml.converter.CMatrixUtil;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.Feature;
import org.jpmml.converter.FieldNameUtil;
import org.jpmml.converter.Label;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.converter.Schema;
import org.jpmml.python.ClassDictUtil;
import sklearn.Estimator;

public class KNeighborsUtil {

	private KNeighborsUtil(){
	}

	static
	public <E extends Estimator & HasNeighbors & HasTrainingData> NearestNeighborModel encodeNeighbors(E estimator, MiningFunction miningFunction, int numberOfInstances, int numberOfFeatures, Schema schema){
		String weights = estimator.getWeights();

		if(!(weights).equals("uniform")){
			throw new IllegalArgumentException(weights);
		}

		List<?> y = estimator.getY();
		List<? extends Number> fitX = estimator.getFitX();

		ClassDictUtil.checkSize(numberOfInstances, y);

		Map<String, List<?>> data = new LinkedHashMap<>();

		InstanceFields instanceFields = new InstanceFields();

		Label label = schema.getLabel();
		if(label != null){
			InstanceField instanceField = new InstanceField(label.getName())
				.setColumn("data:y");

			instanceFields.addInstanceFields(instanceField);

			data.put(instanceField.getColumn(), y);
		}

		KNNInputs knnInputs = new KNNInputs();

		List<? extends Feature> features = schema.getFeatures();
		for(int i = 0; i < features.size(); i++){
			Feature feature = features.get(i);

			ContinuousFeature continuousFeature = feature.toContinuousFeature(estimator.getDataType());

			FieldName name = continuousFeature.getName();

			InstanceField instanceField = new InstanceField(name)
				.setColumn("data:x" + String.valueOf(i + 1));

			instanceFields.addInstanceFields(instanceField);

			KNNInput knnInput = new KNNInput(name);

			knnInputs.addKNNInputs(knnInput);

			data.put(instanceField.getColumn(), CMatrixUtil.getColumn(fitX, numberOfInstances, numberOfFeatures, i));
		}

		TrainingInstances trainingInstances = new TrainingInstances(instanceFields, null, PMMLUtil.createInlineTable(data))
			.setTransformed(true);

		ComparisonMeasure comparisonMeasure = encodeComparisonMeasure(estimator.getMetric(), estimator.getP());

		int numberOfNeighbors = estimator.getNumberOfNeighbors();

		Output output = new Output();

		for(int i = 0; i < numberOfNeighbors; i++){
			int rank = (i + 1);

			OutputField outputField = new OutputField(FieldNameUtil.create("neighbor", rank), OpType.CATEGORICAL, DataType.STRING)
				.setResultFeature(ResultFeature.ENTITY_ID)
				.setRank(rank);

			output.addOutputFields(outputField);
		}

		NearestNeighborModel nearestNeighborModel = new NearestNeighborModel(MiningFunction.REGRESSION, numberOfNeighbors, ModelUtil.createMiningSchema(schema.getLabel()), trainingInstances, comparisonMeasure, knnInputs)
			.setOutput(output);

		return nearestNeighborModel;
	}

	static
	private ComparisonMeasure encodeComparisonMeasure(String metric, int p){
		Measure measure;

		switch(metric){
			case "euclidean":
				{
					measure = new Euclidean();
				}
				break;
			case "manhattan":
				{
					measure = new CityBlock();
				}
				break;
			case "minkowski":
				{
					switch(p){
						case 1:
							measure = new CityBlock();
							break;
						case 2:
							measure = new Euclidean();
							break;
						default:
							measure = new Minkowski(p);
							break;
					}
				}
				break;
			default:
				throw new IllegalArgumentException(metric);
		}

		ComparisonMeasure comparisonMeasure = new ComparisonMeasure(ComparisonMeasure.Kind.DISTANCE, measure)
			.setCompareFunction(CompareFunction.ABS_DIFF);

		return comparisonMeasure;
	}
}
