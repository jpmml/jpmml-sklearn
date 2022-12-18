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
package optbinning;

import java.util.List;
import java.util.Objects;

import org.dmg.pmml.DataType;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.Predicate;
import org.jpmml.converter.CategoricalFeature;
import org.jpmml.converter.PMMLEncoder;

public class BinnedFeature extends CategoricalFeature {

	private List<Predicate> predicates = null;


	public BinnedFeature(PMMLEncoder encoder, DerivedField derivedField, List<?> values, List<Predicate> predicates){
		this(encoder, derivedField.requireName(), derivedField.requireDataType(), values, predicates);
	}

	public BinnedFeature(PMMLEncoder encoder, String name, DataType dataType, List<?> values, List<Predicate> predicates){
		super(encoder, name, dataType, values);

		setPredicates(predicates);
	}

	@Override
	public DerivedField getField(){
		return (DerivedField)super.getField();
	}

	public List<Predicate> getPredicates(){
		return this.predicates;
	}

	private void setPredicates(List<Predicate> predicates){
		predicates = Objects.requireNonNull(predicates);

		if(predicates.isEmpty()){
			throw new IllegalArgumentException();
		}

		this.predicates = predicates;
	}
}