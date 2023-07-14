/*
 * Copyright (c) 2023 Villu Ruusmann
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
package h2o.utils.metaclass;

import java.util.Arrays;
import java.util.Map;

import net.razorvine.pickle.PickleException;
import net.razorvine.pickle.objects.ClassDictConstructor;

public class H2OMetaConstructor extends ClassDictConstructor {

	public H2OMetaConstructor(String module, String name){
		super(module, name);
	}

	@Override
	public Object construct(Object[] args){

		if(args.length != 3){
			throw new PickleException(Arrays.deepToString(args));
		}

		String name = (String)args[0];
		Object[] bases = (Object[])args[1];
		Map<String, ?> dict = (Map<String, ?>)args[2];

		return (ClassDictConstructor)bases[0];
	}
}