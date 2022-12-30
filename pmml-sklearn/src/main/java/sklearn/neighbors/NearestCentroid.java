/*
 * Copyright (c) 2022 Villu Ruusmann
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

import java.util.List;

import org.dmg.pmml.DataType;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.Model;
import org.dmg.pmml.nearest_neighbor.NearestNeighborModel;
import org.jpmml.converter.Schema;
import org.jpmml.python.SliceUtil;
import sklearn.Classifier;

public class NearestCentroid extends Classifier implements HasMetric, HasNumberOfNeighbors, HasTrainingData {

	public NearestCentroid(String module, String name){
		super(module, name);
	}

	@Override
	public int getNumberOfFeatures(){
		int[] shape = getCentroidsShape();

		return shape[1];
	}

	@Override
	public int getNumberOfOutputs(){
		return 1;
	}

	@Override
	public DataType getDataType(){
		return DataType.FLOAT;
	}

	@Override
	public boolean hasProbabilityDistribution(){
		return false;
	}

	@Override
	public Model encodeModel(Schema schema){
		int[] shape = getCentroidsShape();

		int numberOfInstances = shape[0];
		int numberOfFeatures = shape[1];

		NearestNeighborModel nearestNeighborModel = KNeighborsUtil.encodeNeighbors(this, MiningFunction.CLASSIFICATION, numberOfInstances, numberOfFeatures, schema)
			.setCategoricalScoringMethod(NearestNeighborModel.CategoricalScoringMethod.MAJORITY_VOTE);

		return nearestNeighborModel;
	}

	public List<? extends Number> getCentroids(){
		return getNumberArray("centroids_");
	}

	public int[] getCentroidsShape(){
		return getArrayShape("centroids_", 2);
	}

	@Override
	public String getMetric(){
		return getString("metric");
	}

	@Override
	public int getP(){

		// XXX
		if(!containsKey("p")){
			return -1;
		}

		return getInteger("p");
	}

	@Override
	public int getNumberOfNeighbors(){
		return 1;
	}

	@Override
	public List<? extends Number> getFitX(){
		return getCentroids();
	}

	@Override
	public int[] getFitXShape(){
		return getCentroidsShape();
	}

	@Override
	public List<?> getId(){
		return null;
	}

	@Override
	public List<? extends Number> getY(){
		List<?> classes = getClasses();

		return SliceUtil.indices(0, classes.size());
	}

	@Override
	public int[] getYShape(){
		List<?> classes = getClasses();

		return new int[]{classes.size()};
	}
}