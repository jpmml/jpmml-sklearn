/*
 * Copyright (c) 2015 Villu Ruusmann
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
package sklearn.cluster;

import java.util.ArrayList;
import java.util.List;

import com.google.common.base.Function;
import com.google.common.collect.HashMultiset;
import com.google.common.collect.Lists;
import com.google.common.collect.Multiset;
import numpy.core.NDArrayUtil;
import org.dmg.pmml.Array;
import org.dmg.pmml.Cluster;
import org.dmg.pmml.ClusteringField;
import org.dmg.pmml.ClusteringModel;
import org.dmg.pmml.CompareFunctionType;
import org.dmg.pmml.ComparisonMeasure;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.MiningFunctionType;
import org.dmg.pmml.MiningSchema;
import org.dmg.pmml.Output;
import org.dmg.pmml.OutputField;
import org.dmg.pmml.SquaredEuclidean;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.sklearn.ClassDictUtil;
import org.jpmml.sklearn.Schema;
import sklearn.Clusterer;

public class KMeans extends Clusterer {

	public KMeans(String module, String name){
		super(module, name);
	}

	@Override
	public int getNumberOfFeatures(){
		int[] shape = getClusterCentersShape();

		if(shape.length != 2){
			throw new IllegalArgumentException();
		}

		return shape[1];
	}

	@Override
	public ClusteringModel encodeModel(Schema schema){
		int[] shape = getClusterCentersShape();

		int numberOfClusters = shape[0];
		int numberOfFeatures = shape[1];

		List<? extends Number> clusterCenters = getClusterCenters();
		List<? extends Number> labels = getLabels();

		Multiset<Integer> labelCounts = parseLabels(labels);

		List<Cluster> clusters = new ArrayList<>();

		for(int i = 0; i < numberOfClusters; i++){
			Array array = encodeArray(NDArrayUtil.getRow(clusterCenters, numberOfClusters, numberOfFeatures, i));

			Cluster cluster = new Cluster()
				.setId(String.valueOf(i))
				.setSize((labelCounts.size () > 0 ? labelCounts.count(i) : null))
				.setArray(array);

			clusters.add(cluster);
		}

		List<FieldName> activeFields = schema.getActiveFields();

		Function<FieldName, ClusteringField> function = new Function<FieldName, ClusteringField>(){

			@Override
			public ClusteringField apply(FieldName name){
				ClusteringField clusteringField = new ClusteringField(name);

				return clusteringField;
			}
		};

		List<ClusteringField> clusteringFields = new ArrayList<>(Lists.transform(activeFields, function));

		ComparisonMeasure comparisonMeasure = new ComparisonMeasure(ComparisonMeasure.Kind.DISTANCE)
			.setCompareFunction(CompareFunctionType.ABS_DIFF)
			.setMeasure(new SquaredEuclidean());

		MiningSchema miningSchema = PMMLUtil.createMiningSchema(null, activeFields);

		List<OutputField> outputFields = new ArrayList<>();
		outputFields.add(PMMLUtil.createPredictedField(FieldName.create("Cluster")));
		outputFields.addAll(encodeAffinityFields(clusters));

		Output output = new Output(outputFields);

		ClusteringModel clusteringModel = new ClusteringModel(MiningFunctionType.CLUSTERING, ClusteringModel.ModelClass.CENTER_BASED, numberOfClusters, miningSchema, comparisonMeasure, clusteringFields, clusters)
			.setOutput(output);

		return clusteringModel;
	}

	public List<? extends Number> getClusterCenters(){
		return (List)ClassDictUtil.getArray(this, "cluster_centers_");
	}

	public List<? extends Number> getLabels(){
		return (List)ClassDictUtil.getArray(this, "labels_");
	}

	private int[] getClusterCentersShape(){
		return ClassDictUtil.getShape(this, "cluster_centers_");
	}

	static
	private Multiset<Integer> parseLabels(List<? extends Number> labels){
		Multiset<Integer> result = HashMultiset.create();

		if(labels != null){
			Function<Number, Integer> function = new Function<Number, Integer>(){

				@Override
				public Integer apply(Number number){

					if(number instanceof Integer){
						return (Integer)number;
					}

					return number.intValue();
				}
			};

			result.addAll(Lists.transform(labels, function));
		}

		return result;
	}

	static
	private Array encodeArray(List<? extends Number> values){
		Function<Number, String> function = new Function<Number, String>(){

			@Override
			public String apply(Number number){
				return PMMLUtil.formatValue(number);
			}
		};

		String value = PMMLUtil.formatArrayValue(Lists.transform(values, function));

		Array array = new Array(Array.Type.REAL, value);

		return array;
	}

	static
	private List<OutputField> encodeAffinityFields(List<Cluster> clusters){
		Function<Cluster, OutputField> function = new Function<Cluster, OutputField>(){

			@Override
			public OutputField apply(Cluster cluster){
				OutputField outputField = PMMLUtil.createAffinityField(cluster.getId());

				return outputField;
			}
		};

		return Lists.transform(clusters, function);
	}
}