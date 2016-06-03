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

import com.google.common.collect.HashMultiset;
import com.google.common.collect.Multiset;
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
import org.dmg.pmml.SquaredEuclidean;
import org.jpmml.converter.ClusteringModelUtil;
import org.jpmml.converter.Feature;
import org.jpmml.converter.FeatureSchema;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.converter.ValueUtil;
import org.jpmml.sklearn.ClassDictUtil;
import org.jpmml.sklearn.MatrixUtil;
import sklearn.Clusterer;

public class KMeans extends Clusterer {

	public KMeans(String module, String name){
		super(module, name);
	}

	@Override
	public int getNumberOfFeatures(){
		int[] shape = getClusterCentersShape();

		return shape[1];
	}

	@Override
	public ClusteringModel encodeModel(FeatureSchema schema){
		int[] shape = getClusterCentersShape();

		int numberOfClusters = shape[0];
		int numberOfFeatures = shape[1];

		List<? extends Number> clusterCenters = getClusterCenters();
		List<Integer> labels = getLabels();

		Multiset<Integer> labelCounts = HashMultiset.create();

		if(labels != null){
			labelCounts.addAll(labels);
		}

		List<Cluster> clusters = new ArrayList<>();

		for(int i = 0; i < numberOfClusters; i++){
			Array array = PMMLUtil.createRealArray(MatrixUtil.getRow(clusterCenters, numberOfClusters, numberOfFeatures, i));

			Cluster cluster = new Cluster()
				.setId(String.valueOf(i))
				.setSize((labelCounts.size () > 0 ? labelCounts.count(i) : null))
				.setArray(array);

			clusters.add(cluster);
		}

		List<FieldName> names = new ArrayList<>();

		List<Feature> features = schema.getFeatures();
		for(Feature feature : features){
			names.add(feature.getName());
		}

		List<ClusteringField> clusteringFields = ClusteringModelUtil.createClusteringFields(names);

		ComparisonMeasure comparisonMeasure = new ComparisonMeasure(ComparisonMeasure.Kind.DISTANCE)
			.setCompareFunction(CompareFunctionType.ABS_DIFF)
			.setMeasure(new SquaredEuclidean());

		MiningSchema miningSchema = ModelUtil.createMiningSchema(schema);

		Output output = ClusteringModelUtil.createOutput(FieldName.create("Cluster"), clusters);

		ClusteringModel clusteringModel = new ClusteringModel(MiningFunctionType.CLUSTERING, ClusteringModel.ModelClass.CENTER_BASED, numberOfClusters, miningSchema, comparisonMeasure, clusteringFields, clusters)
			.setOutput(output);

		return clusteringModel;
	}

	public List<? extends Number> getClusterCenters(){
		return (List)ClassDictUtil.getArray(this, "cluster_centers_");
	}

	public List<Integer> getLabels(){
		return ValueUtil.asIntegers((List)ClassDictUtil.getArray(this, "labels_"));
	}

	private int[] getClusterCentersShape(){
		return ClassDictUtil.getShape(this, "cluster_centers_", 2);
	}
}