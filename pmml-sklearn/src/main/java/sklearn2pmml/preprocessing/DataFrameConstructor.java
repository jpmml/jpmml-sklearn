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
package sklearn2pmml.preprocessing;

import java.util.ArrayList;
import java.util.List;

import org.dmg.pmml.DataField;
import org.dmg.pmml.DataType;
import org.dmg.pmml.OpType;
import org.jpmml.converter.Feature;
import org.jpmml.converter.SchemaUtil;
import org.jpmml.converter.TypeUtil;
import org.jpmml.converter.WildcardFeature;
import org.jpmml.python.TypeInfo;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn.Initializer;

public class DataFrameConstructor extends Initializer {

	public DataFrameConstructor(String module, String name){
		super(module, name);
	}

	@Override
	public List<Feature> initializeFeatures(SkLearnEncoder encoder){
		List<String> columns = getColumns();
		TypeInfo dtype = getDType();

		DataType dataType = dtype.getDataType();
		OpType opType = TypeUtil.getOpType(dataType);

		List<Feature> result = new ArrayList<>();

		for(String column : columns){
			DataField dataField = encoder.createDataField(column, opType, dataType);

			result.add(new WildcardFeature(encoder, dataField));
		}

		return result;
	}

	@Override
	public List<Feature> encodeFeatures(List<Feature> features, SkLearnEncoder encoder){

		if(features.isEmpty()){
			return initializeFeatures(encoder);
		}

		List<String> columns = getColumns();
		TypeInfo dtype = getDType();

		DataType dataType = dtype.getDataType();

		SchemaUtil.checkSize(columns.size(), features);

		List<Feature> result = new ArrayList<>();

		for(int i = 0; i < features.size(); i++){
			Feature feature = features.get(i);
			String column = columns.get(i);

			if(feature.getDataType() != dataType){
				throw new IllegalArgumentException();
			}

			encoder.renameFeature(feature, column);

			result.add(feature);
		}

		return result;
	}

	public List<String> getColumns(){
		return getList("columns", String.class);
	}

	public TypeInfo getDType(){
		return getDType("dtype", true);
	}
}