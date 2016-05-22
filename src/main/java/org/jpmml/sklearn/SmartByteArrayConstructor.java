/*
 * Copyright (c) 2016 Villu Ruusmann
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

import net.razorvine.pickle.PickleException;
import net.razorvine.pickle.objects.ByteArrayConstructor;

public class SmartByteArrayConstructor extends ByteArrayConstructor {

	@Override
	public Object construct(Object[] args) throws PickleException {

		if(args.length == 1){
			Object value = args[0];

			if(value instanceof byte[]){
				return value;
			}
		}

		return super.construct(args);
	}
}