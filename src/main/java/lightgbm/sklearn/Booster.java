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

import java.io.IOException;
import java.io.StringReader;
import java.util.List;

import com.google.common.io.CharStreams;
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
		String handle = getHandle();

		try(StringReader reader = new StringReader(handle)){
			List<String> lines = CharStreams.readLines(reader);

			return LightGBMUtil.loadGBDT(lines.iterator());
		} catch(IOException ioe){
			throw new RuntimeException(ioe);
		}
	}

	public String getHandle(){
		return (String)get("handle");
	}
}