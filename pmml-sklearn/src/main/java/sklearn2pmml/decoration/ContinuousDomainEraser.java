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

import java.util.List;

import org.dmg.pmml.Field;
import org.dmg.pmml.HasContinuousDomain;
import org.dmg.pmml.Interval;

public class ContinuousDomainEraser extends DomainEraser {

	public ContinuousDomainEraser(String module, String name){
		super(module, name);
	}

	@Override
	public void clear(Field<?> field){

		if(field instanceof HasContinuousDomain){
			HasContinuousDomain<?> hasContinuousDomain = (HasContinuousDomain<?>)field;

			if(hasContinuousDomain.hasIntervals()){
				List<Interval> intervals = hasContinuousDomain.getIntervals();

				intervals.clear();
			}
		}
	}
}