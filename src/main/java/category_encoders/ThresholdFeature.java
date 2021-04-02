/*
 * Copyright (c) 2021 Villu Ruusmann
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
package category_encoders;

import java.util.Map;
import java.util.Objects;

import org.dmg.pmml.DerivedField;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.Feature;
import org.jpmml.converter.PMMLEncoder;
import org.jpmml.model.ToStringHelper;

public class ThresholdFeature extends Feature {

	private Map<?, ? extends Number> mapping = null;


	public ThresholdFeature(PMMLEncoder encoder, DerivedField derivedField, Map<?, ? extends Number> mapping){
		super(encoder, derivedField.getName(), derivedField.getDataType());

		setMapping(mapping);
	}

	@Override
	public ContinuousFeature toContinuousFeature(){
		PMMLEncoder encoder = getEncoder();

		DerivedField derivedField = (DerivedField)encoder.toContinuous(getName());

		return new ContinuousFeature(encoder, derivedField);
	}

	@Override
	protected ToStringHelper toStringHelper(){
		return new ToStringHelper(this)
			.add("mapping", getMapping());
	}

	public Map<?, ? extends Number> getMapping(){
		return this.mapping;
	}

	private void setMapping(Map<?, ? extends Number> mapping){
		this.mapping = Objects.requireNonNull(mapping);
	}
}