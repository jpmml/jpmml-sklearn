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
package types;

import java.util.Arrays;
import java.util.Map;

import net.razorvine.pickle.PickleException;
import net.razorvine.pickle.objects.ClassDict;

public class FunctionType extends ClassDict {

	public FunctionType(){
		this("types", "FunctionType");
	}

	public FunctionType(String module, String name){
		super(module, name);
	}

	public void __setstate__(Object[] args){

		if(args.length != 2){
			throw new PickleException(Arrays.deepToString(args));
		}

		// XXX
		Map<String, ?> dict = (Map<String, ?>)args[0];
		Map<String, ?> kwdefaults = (Map<String, ?>)args[1];
	}
}