/*
 * Copyright (c) 2026 Villu Ruusmann
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
package org.jpmml.sklearn;

import java.util.List;

import org.dmg.pmml.Field;
import org.jpmml.converter.CategoricalFeature;
import org.jpmml.converter.Feature;
import org.jpmml.converter.OrdinalFeature;
import org.jpmml.converter.WildcardFeature;
import org.jpmml.python.HasArray;
import org.jpmml.python.TypeInfo;
import pandas.core.CategoricalDtype;
import polars.datatypes.Categorical;
import polars.datatypes.Enum;

public class DTypeUtil {

	private DTypeUtil(){
	}

	static
	public List<?> getCategories(TypeInfo dtype){

		if(dtype instanceof CategoricalDtype){
			CategoricalDtype categoricalDtype = (CategoricalDtype)dtype;

			return categoricalDtype.getValues();
		} else

		if(dtype instanceof Enum){
			Enum _enum = (Enum)dtype;

			HasArray categories = _enum.getCategories();

			return categories.getArrayContent();
		}

		return null;
	}

	static
	public Feature refineFeature(Feature feature, TypeInfo dtype, SkLearnEncoder encoder){

		if(dtype instanceof CategoricalDtype){
			CategoricalDtype categoricalDtype = (CategoricalDtype)dtype;

			Boolean ordered = categoricalDtype.getOrdered();
			List<?> values = categoricalDtype.getValues();

			return refineFeature(feature, ordered, values, encoder);
		} else

		if(dtype instanceof Categorical){
			Categorical categorical = (Categorical)dtype;

			return refineFeature(feature, false, null, encoder);
		} else

		if(dtype instanceof Enum){
			Enum _enum = (Enum)dtype;

			HasArray categories = _enum.getCategories();

			return refineFeature(feature, true, categories.getArrayContent(), encoder);
		}

		return feature;
	}

	static
	public Feature refineFeature(Feature feature, boolean ordered, List<?> values, SkLearnEncoder encoder){

		if(feature instanceof WildcardFeature){
			WildcardFeature wildcardFeature = (WildcardFeature)feature;

			if(ordered){
				return wildcardFeature.toOrdinalFeature(values);
			} else

			{
				return wildcardFeature.toCategoricalFeature(values);
			}
		} else

		{
			String name = feature.getName();

			if(ordered){
				Field<?> field = encoder.toOrdinal(name, values);

				return new OrdinalFeature(encoder, field, values);
			} else

			{
				Field<?> field = encoder.toCategorical(name, values);

				return new CategoricalFeature(encoder, field, values);
			}
		}
	}
}