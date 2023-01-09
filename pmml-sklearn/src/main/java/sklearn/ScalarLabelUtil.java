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
package sklearn;

import java.util.List;
import java.util.Objects;

import org.dmg.pmml.Field;
import org.dmg.pmml.OpType;
import org.jpmml.converter.CategoricalFeature;
import org.jpmml.converter.CategoricalLabel;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.ContinuousLabel;
import org.jpmml.converter.Feature;
import org.jpmml.converter.PMMLEncoder;
import org.jpmml.converter.ScalarLabel;

public class ScalarLabelUtil {

	private ScalarLabelUtil(){
	}

	static
	public OpType getOpType(ScalarLabel scalarLabel){

		if(scalarLabel instanceof CategoricalLabel){
			CategoricalLabel categoricalLabel = (CategoricalLabel)scalarLabel;

			return OpType.CATEGORICAL;
		} else

		if(scalarLabel instanceof ContinuousLabel){
			ContinuousLabel continuousLabel = (ContinuousLabel)scalarLabel;

			return OpType.CONTINUOUS;
		} else

		{
			throw new IllegalArgumentException();
		}
	}

	static
	public Feature toFeature(ScalarLabel scalarLabel, PMMLEncoder encoder){
		Field<?> field = encoder.getField(scalarLabel.getName());

		return toFeature(scalarLabel, field, encoder);
	}

	static
	public Feature toFeature(ScalarLabel scalarLabel, Field<?> field, PMMLEncoder encoder){

		if(scalarLabel instanceof ContinuousLabel){
			ContinuousLabel continuousLabel = (ContinuousLabel)scalarLabel;

			return new ContinuousFeature(encoder, field);
		} else

		if(scalarLabel instanceof CategoricalLabel){
			CategoricalLabel categoricalLabel = (CategoricalLabel)scalarLabel;

			return new CategoricalFeature(encoder, field, categoricalLabel.getValues());
		} else

		{
			throw new IllegalArgumentException();
		}
	}

	static
	public Feature findLabelFeature(ScalarLabel scalarLabel, List<? extends Feature> features){
		String name = scalarLabel.getName();

		for(Feature feature : features){

			if(Objects.equals(feature.getName(), name)){
				return feature;
			}
		}

		return null;
	}
}