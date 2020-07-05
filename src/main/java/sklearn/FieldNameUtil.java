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
package sklearn;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import org.dmg.pmml.FieldName;
import org.jpmml.converter.Feature;

public class FieldNameUtil {

	private FieldNameUtil(){
	}

	static
	public FieldName create(String function, Object... args){
		return create(function, Arrays.asList(args));
	}

	static
	public FieldName create(String function, List<?> args){
		String name = format(function, args);

		return FieldName.create(name);
	}

	static
	public FieldName create(FieldName name, int index){

		if(index < 0){
			throw new IllegalArgumentException();
		}

		return FieldName.create(name.getValue() + "[" + index + "]");
	}

	static
	private String format(String function, List<?> args){

		if(args == null || args.isEmpty()){
			return function;
		} else

		{
			Stream<?> argStream;

			if(args.size() <= 5){
				argStream = args.stream();
			} else

			{
				argStream = Stream.of(
					args.subList(0, 2).stream(),
					Stream.of(".."),
					args.subList(args.size() - 2, args.size()).stream()
				).flatMap(x -> x);
			}

			return argStream
				.map(FieldNameUtil::toString)
				.collect(Collectors.joining(", ", function + "(", ")"));
		}
	}

	static
	private String toString(Object object){

		if(object instanceof Feature){
			Feature feature = (Feature)object;

			object = feature.getName();
		}

		return String.valueOf(object);
	}
}