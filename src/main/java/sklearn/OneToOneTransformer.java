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
public class OneToOneTransformer extends Transformer {

	public OneToOneTransformer(String module, String name){
		super(module, name);
	}

	abstract
	public Expression encode(FieldName name);

	@Override
	public int getNumberOfInputs(){
		return 1;
	}

	@Override
	public int getNumberOfOutputs(){
		return 1;
	}

	@Override
	public Expression encode(int index, List<FieldName> names){

		if(index != 0 || names.size() != 1){
			throw new IllegalArgumentException();
		}

		return encode(names.get(0));
	}
}