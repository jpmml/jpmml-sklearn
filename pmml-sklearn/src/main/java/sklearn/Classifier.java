/*
 * Copyright (c) 2015 Villu Ruusmann
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
import java.util.stream.Collectors;

import org.dmg.pmml.MiningFunction;

abstract
public class Classifier extends Estimator implements HasClasses {

	public Classifier(String module, String name){
		super(module, name);
	}

	@Override
	public MiningFunction getMiningFunction(){
		return MiningFunction.CLASSIFICATION;
	}

	public boolean hasProbabilityDistribution(){
		return true;
	}

	@Override
	public List<?> getClasses(){
		List<?> values = getArray(SkLearnFields.CLASSES);

		return values.stream()
			.map(value -> (value instanceof Long) ? Math.toIntExact((Long)value) : value)
			.collect(Collectors.toList());
	}

	public static final String FIELD_PROBABILITY = "probability";
}