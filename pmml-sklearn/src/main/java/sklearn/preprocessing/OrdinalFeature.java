/*
 * Copyright (c) 2025 Villu Ruusmann
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
package sklearn.preprocessing;

import java.util.List;

import org.dmg.pmml.DataType;
import org.dmg.pmml.Field;
import org.jpmml.converter.CategoricalFeature;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.Feature;
import org.jpmml.converter.IndexFeature;
import org.jpmml.converter.PMMLEncoder;

abstract
public class OrdinalFeature extends CategoricalFeature {

	public OrdinalFeature(PMMLEncoder encoder, Field<?> field, List<?> values){
		super(encoder, field, values);
	}

	public OrdinalFeature(PMMLEncoder encoder, Feature feature, List<?> values){
		super(encoder, feature, values);
	}

	public OrdinalFeature(PMMLEncoder encoder, String name, DataType dataType, List<?> values){
		super(encoder, name, dataType, values);
	}

	abstract
	public IndexFeature getEncodedFeature();

	@Override
	public ContinuousFeature toContinuousFeature(){
		IndexFeature encodedFeature = getEncodedFeature();

		return encodedFeature.toContinuousFeature();
	}
}