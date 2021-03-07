/*
 * Copyright (c) 2021 Villu Ruusmann
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
package sklearn2pmml.decoration;

import java.util.Iterator;
import java.util.List;

import org.dmg.pmml.Field;
import org.dmg.pmml.HasDiscreteDomain;
import org.dmg.pmml.Value;

public class DiscreteDomainEraser extends DomainEraser {

	public DiscreteDomainEraser(String module, String name){
		super(module, name);
	}

	@Override
	public void clear(Field<?> field){

		if(field instanceof HasDiscreteDomain){
			HasDiscreteDomain<?> hasDiscreteDomain = (HasDiscreteDomain<?>)field;

			if(hasDiscreteDomain.hasValues()){
				List<Value> values = hasDiscreteDomain.getValues();

				for(Iterator<Value> it = values.iterator(); it.hasNext(); ){
					Value value = it.next();

					Value.Property property = value.getProperty();
					switch(property){
						case VALID:
							it.remove();
							break;
						case INVALID:
						case MISSING:
							break;
						default:
							break;
					}
				}
			}
		}
	}
}