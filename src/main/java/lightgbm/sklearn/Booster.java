/*
 * Copyright (c) 2017 Villu Ruusmann
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
package lightgbm.sklearn;

import java.util.List;

import com.google.common.base.Function;
import com.google.common.collect.Iterators;
import net.razorvine.pickle.objects.ClassDict;
import org.jpmml.lightgbm.GBDT;
import org.jpmml.lightgbm.LightGBMUtil;

public class Booster extends ClassDict {

	private GBDT gbdt = null;


	public Booster(String module, String name){
		super(module, name);
	}

	public GBDT getGBDT(){

		if(this.gbdt == null){
			this.gbdt = loadGBDT();
		}

		return this.gbdt;
	}

	private GBDT loadGBDT(){
		List<String> handle = getHandle();

		Function<String, String> function = new Function<String, String>(){

			@Override
			public String apply(String string){
				return string.trim(); // XXX
			}
		};

		return LightGBMUtil.loadGBDT(Iterators.transform(handle.iterator(), function));
	}

	public List<String> getHandle(){
		return (List)get("handle");
	}
}