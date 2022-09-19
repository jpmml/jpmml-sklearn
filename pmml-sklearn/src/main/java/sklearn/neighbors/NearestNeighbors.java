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
import org.dmg.pmml.nearest_neighbor.NearestNeighborModel;
import org.jpmml.converter.Schema;
import sklearn.Clusterer;

public class NearestNeighbors extends Clusterer implements HasMetric, HasNumberOfNeighbors, HasTrainingData {

	public NearestNeighbors(String module, String name){
		super(module, name);
	}

	@Override
	public int getNumberOfFeatures(){
		int[] shape = getFitXShape();

		return shape[1];
	}

	@Override
	public int getNumberOfOutputs(){
		return 0;
	}

	@Override
	public DataType getDataType(){
		return DataType.FLOAT;
	}

	@Override
	public NearestNeighborModel encodeModel(Schema schema){
		int[] shape = getFitXShape();

		int numberOfInstances = shape[0];
		int numberOfFeatures = shape[1];

		return KNeighborsUtil.encodeNeighbors(this, MiningFunction.CLUSTERING, numberOfInstances, numberOfFeatures, schema);
	}

	@Override
	public String getMetric(){
		return getString("metric");
	}

	@Override
	public int getP(){
		return getInteger("p");
	}

	@Override
	public int getNumberOfNeighbors(){
		return getInteger("n_neighbors");
	}

	@Override
	public List<? extends Number> getFitX(){
		return getNumberArray("_fit_X");
	}

	@Override
	public int[] getFitXShape(){
		return getArrayShape("_fit_X", 2);
	}

	@Override
	public List<?> getId(){

		// XXX
		if(!containsKey("_id")){
			int[] shape = getFitXShape();

			return KNeighborsUtil.createRange(0, shape[0]);
		}

		return getArray("_id");
	}

	@Override
	public List<? extends Number> getY(){
		return null;
	}

	@Override
	public int[] getYShape(){
		throw new UnsupportedOperationException();
	}
}