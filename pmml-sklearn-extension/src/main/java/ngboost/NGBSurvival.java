/*
 * Copyright (c) 2026 Villu Ruusmann
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
package ngboost;

import java.util.function.Function;

import net.razorvine.pickle.objects.ClassDictConstructor;
import org.jpmml.python.ClassDictConstructorUtil;

public class NGBSurvival extends NGBRegressor {

	public NGBSurvival(String module, String name){
		super(module, name);
	}

	public ClassDictConstructor getBaseDist(){
		return get("_basedist", ClassDictConstructor.class);
	}

	@Override
	protected Function<String, String> getDistNameFunction(){
		Function<String, String> function = new Function<String, String>(){

			@Override
			public String apply(String name){
				ClassDictConstructor baseDist = getBaseDist();

				return ClassDictConstructorUtil.getName(baseDist);
			}
		};

		return function;
	}
}