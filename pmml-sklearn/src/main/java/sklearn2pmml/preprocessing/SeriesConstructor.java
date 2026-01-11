/*
 * Copyright (c) 2024 Villu Ruusmann
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

import java.util.Collections;
import java.util.List;
import java.util.Objects;

import org.dmg.pmml.DataType;
import org.jpmml.converter.Feature;
import org.jpmml.converter.SchemaUtil;
import org.jpmml.python.TypeInfo;
import org.jpmml.sklearn.SkLearnEncoder;
import pandas.CategoricalDtypeUtil;
import pandas.core.CategoricalDtype;
import sklearn.Transformer;

public class SeriesConstructor extends Transformer {

	public SeriesConstructor(String module, String name){
		super(module, name);
	}

	@Override
	public List<Feature> encodeFeatures(List<Feature> features, SkLearnEncoder encoder){
		TypeInfo dtype = getDType();
		String name = getName();

		DataType dataType = dtype.getDataType();

		Feature feature = SchemaUtil.getOnlyFeature(features)
			.expectDataType(dataType);

		if(dtype instanceof CategoricalDtype){
			CategoricalDtype categoricalDtype = (CategoricalDtype)dtype;

			feature = CategoricalDtypeUtil.refineFeature(feature, categoricalDtype, encoder);
		} // End if

		if(name != null && !Objects.equals(feature.getName(), name)){
			encoder.renameFeature(feature, name);
		}

		return Collections.singletonList(feature);
	}

	public TypeInfo getDType(){
		return getDType("dtype");
	}

	public String getName(){
		return getOptionalString("name");
	}
}