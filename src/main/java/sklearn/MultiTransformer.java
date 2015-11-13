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

import org.dmg.pmml.Expression;
import org.dmg.pmml.FieldName;

abstract
public class MultiTransformer extends Transformer {

	public MultiTransformer(String module, String name){
		super(module, name);
	}

	abstract
	public int getNumberOfFeatures();

	abstract
	public Expression encode(int index, FieldName name);

	@Override
	public int getNumberOfInputs(){
		return getNumberOfFeatures();
	}

	@Override
	public int getNumberOfOutputs(){
		return getNumberOfFeatures();
	}

	@Override
	public Expression encode(int index, List<FieldName> names){
		int numberOfFeatures = getNumberOfFeatures();

		if(names.size() != numberOfFeatures){
			throw new IllegalArgumentException();
		}

		return encode(index, names.get(index));
	}
}