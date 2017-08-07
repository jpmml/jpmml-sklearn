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
import org.dmg.pmml.CompareFunction;
import org.dmg.pmml.ComparisonMeasure;
import org.dmg.pmml.DataType;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.SquaredEuclidean;
import org.dmg.pmml.clustering.Cluster;
import org.dmg.pmml.clustering.ClusteringModel;
import org.jpmml.converter.CMatrixUtil;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.converter.Schema;
import org.jpmml.converter.ValueUtil;
import org.jpmml.converter.clustering.ClusteringModelUtil;
import org.jpmml.sklearn.ClassDictUtil;
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
	public ClusteringModel encodeModel(Schema schema){
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
			Cluster cluster = new Cluster()
				.setId(String.valueOf(i))
				.setSize((labelCounts.size () > 0 ? labelCounts.count(i) : null))
				.setArray(PMMLUtil.createRealArray(CMatrixUtil.getRow(clusterCenters, numberOfClusters, numberOfFeatures, i)));

			clusters.add(cluster);
		}

		ComparisonMeasure comparisonMeasure = new ComparisonMeasure(ComparisonMeasure.Kind.DISTANCE)
			.setCompareFunction(CompareFunction.ABS_DIFF)
			.setMeasure(new SquaredEuclidean());

		ClusteringModel clusteringModel = new ClusteringModel(MiningFunction.CLUSTERING, ClusteringModel.ModelClass.CENTER_BASED, numberOfClusters, ModelUtil.createMiningSchema(schema.getLabel()), comparisonMeasure, ClusteringModelUtil.createClusteringFields(schema.getFeatures()), clusters)
			.setOutput(ClusteringModelUtil.createOutput(FieldName.create("Cluster"), DataType.DOUBLE, clusters));

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