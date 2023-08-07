/*
 * Copyright (c) 2018 Villu Ruusmann
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
package sklearn2pmml.preprocessing.h2o;

import java.util.List;

import org.dmg.pmml.DataType;
import org.dmg.pmml.OpType;
import org.jpmml.converter.Feature;
import org.jpmml.converter.FeatureList;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn.Transformer;

public class H2OFrameConstructor extends Transformer {

	public H2OFrameConstructor(String module, String name){
		super(module, name);
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
		List<String> columnNames = getColumnNames();

		if(columnNames != null){
			FeatureList namedFeatures = new FeatureList(features, columnNames);

			return namedFeatures;
		}

		return features;
	}

	public List<String> getColumnNames(){
		Object columnNames = getOptionalObject("column_names");

		if(columnNames != null){
			return getListLike("column_names", String.class);
		}

		return null;
	}
}