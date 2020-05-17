/*
 * Copyright (c) 2020 Villu Ruusmann
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

import java.util.Arrays;

import net.razorvine.pickle.PickleException;
import net.razorvine.pickle.objects.ClassDict;

public class NamedTupleConstructor extends ObjectConstructor {

	public NamedTupleConstructor(String module, String name, Class<? extends NamedTuple> clazz){
		super(module, name, clazz);
	}

	@Override
	public ClassDict construct(Object[] args){
		NamedTuple dict = (NamedTuple)newObject();

		String[] names = dict.names();

		if(names.length != args.length){
			throw new PickleException(Arrays.deepToString(args));
		}

		for(int i = 0; i < names.length; i++){
			dict.put(names[i], args[i]);
		}

		return dict;
	}
}