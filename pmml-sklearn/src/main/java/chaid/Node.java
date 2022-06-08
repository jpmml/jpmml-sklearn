/*
 * Copyright (c) 2022 Villu Ruusmann
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
package chaid;

import java.util.List;

import org.jpmml.python.PythonObject;

public class Node extends PythonObject {

	public Node(String module, String name){
		super(module, name);
	}

	public Column getDepV(){
		return get("dep_v", Column.class);
	}

	public List<Integer> getIndices(){
		return getIntegerArray("indices");
	}

	public Split getSplit(){
		return get("split", Split.class);
	}
}