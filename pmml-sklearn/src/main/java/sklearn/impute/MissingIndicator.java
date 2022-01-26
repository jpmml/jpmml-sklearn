/*
 * Copyright (c) 2019 Villu Ruusmann
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
package sklearn.impute;

import java.util.ArrayList;
import java.util.List;

import org.dmg.pmml.DataType;
import org.dmg.pmml.OpType;
import org.jpmml.converter.Feature;
import org.jpmml.converter.ValueUtil;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn.Transformer;

public class MissingIndicator extends Transformer {

	public MissingIndicator(String module, String name){
		super(module, name);
	}

	@Override
	public int getNumberOfFeatures(){
		int[] shape = getFeatureIndicesShape();

		return shape[0];
	}

	@Override
	public OpType getOpType(){
		throw new UnsupportedOperationException();
	}

	@Override
	public DataType getDataType(){
		throw new UnsupportedOperationException();
	}

	@Override
	public List<Feature> encodeFeatures(List<Feature> features, SkLearnEncoder encoder){
		List<Integer> featureIndices = getFeatureIndices();
		Object missingValues = getMissingValues();

		if(ValueUtil.isNaN(missingValues)){
			missingValues = null;
		}

		List<Feature> result = new ArrayList<>();

		for(Integer featureIndex : featureIndices){
			Feature feature = features.get(featureIndex);

			feature = ImputerUtil.encodeIndicatorFeature(this, feature, missingValues, encoder);

			result.add(feature);
		}

		return result;
	}

	public List<Integer> getFeatureIndices(){
		return getIntegerArray("features_");
	}

	public int[] getFeatureIndicesShape(){
		return getArrayShape("features_", 1);
	}

	public Object getMissingValues(){
		return getOptionalScalar("missing_values");
	}
}