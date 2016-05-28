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

import java.util.ArrayList;
import java.util.List;

import javax.xml.parsers.DocumentBuilder;

import org.dmg.pmml.CityBlock;
import org.dmg.pmml.CompareFunctionType;
import org.dmg.pmml.ComparisonMeasure;
import org.dmg.pmml.Euclidean;
import org.dmg.pmml.FeatureType;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.InlineTable;
import org.dmg.pmml.InstanceField;
import org.dmg.pmml.InstanceFields;
import org.dmg.pmml.KNNInput;
import org.dmg.pmml.KNNInputs;
import org.dmg.pmml.Measure;
import org.dmg.pmml.MiningFunctionType;
import org.dmg.pmml.MiningSchema;
import org.dmg.pmml.Minkowski;
import org.dmg.pmml.NearestNeighborModel;
import org.dmg.pmml.Output;
import org.dmg.pmml.OutputField;
import org.dmg.pmml.Row;
import org.dmg.pmml.TrainingInstances;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.Schema;
import org.jpmml.sklearn.DOMUtil;
import org.jpmml.sklearn.MatrixUtil;
import sklearn.Estimator;

public class KNeighborsUtil {

	private KNeighborsUtil(){
	}

	static
	public <E extends Estimator & HasNeighbors & HasTrainingData> NearestNeighborModel encodeNeighbors(E estimator, MiningFunctionType miningFunction, int numberOfInstances, int numberOfFeatures, Schema schema){
		List<String> keys = new ArrayList<>();

		InstanceFields instanceFields = new InstanceFields();

		KNNInputs knnInputs = new KNNInputs();

		FieldName targetField = schema.getTargetField();
		if(targetField != null){
			InstanceField instanceField = new InstanceField(targetField)
				.setColumn(targetField.getValue());

			instanceFields.addInstanceFields(instanceField);

			keys.add(instanceField.getColumn());
		}

		List<FieldName> activeFields = schema.getActiveFields();
		for(FieldName activeField : activeFields){
			InstanceField instanceField = new InstanceField(activeField)
				.setColumn(activeField.getValue());

			instanceFields.addInstanceFields(instanceField);

			keys.add(instanceField.getColumn());

			KNNInput knnInput = new KNNInput(activeField);

			knnInputs.addKNNInputs(knnInput);
		}

		DocumentBuilder documentBuilder = DOMUtil.createDocumentBuilder();

		InlineTable inlineTable = new InlineTable();

		List<?> y = estimator.getY();
		if(y.size() != numberOfInstances){
			throw new IllegalArgumentException();
		}

		List<? extends Number> fitX = estimator.getFitX();

		for(int i = 0; i < numberOfInstances; i++){
			List<Object> values = new ArrayList<>(1 + numberOfFeatures);
			values.add(y.get(i));
			values.addAll(MatrixUtil.getRow(fitX, numberOfInstances, numberOfFeatures, i));

			Row row = DOMUtil.createRow(documentBuilder, keys, values);

			inlineTable.addRows(row);
		}

		TrainingInstances trainingInstances = new TrainingInstances(instanceFields)
			.setInlineTable(inlineTable)
			.setTransformed(true);

		ComparisonMeasure comparisonMeasure = encodeComparisonMeasure(estimator.getMetric(), estimator.getP());

		String weights = estimator.getWeights();
		if(!(weights).equals("uniform")){
			throw new IllegalArgumentException(weights);
		}

		int numberOfNeighbors = estimator.getNumberOfNeighbors();

		List<OutputField> outputFields = new ArrayList<>(numberOfNeighbors);

		for(int i = 0; i < numberOfNeighbors; i++){
			int rank = (i + 1);

			OutputField outputField = new OutputField(FieldName.create("neighbor_" + rank))
				.setFeature(FeatureType.ENTITY_ID)
				.setRank(rank);

			outputFields.add(outputField);
		}

		Output output = new Output(outputFields);

		MiningSchema miningSchema = ModelUtil.createMiningSchema(schema);

		NearestNeighborModel nearestNeighborModel = new NearestNeighborModel(MiningFunctionType.REGRESSION, numberOfNeighbors, miningSchema, trainingInstances, comparisonMeasure, knnInputs)
			.setOutput(output);

		return nearestNeighborModel;
	}

	static
	private ComparisonMeasure encodeComparisonMeasure(String metric, int p){

		if(("minkowski").equals(metric)){
			Measure measure;

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

			ComparisonMeasure comparisonMeasure = new ComparisonMeasure(ComparisonMeasure.Kind.DISTANCE)
				.setCompareFunction(CompareFunctionType.ABS_DIFF)
				.setMeasure(measure);

			return comparisonMeasure;
		} else

		{
			throw new IllegalArgumentException(metric);
		}
	}
}