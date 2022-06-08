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
package treelib;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;

import org.jpmml.python.PythonObject;

public class Tree extends PythonObject {

	public Tree(String module, String name){
		super(module, name);
	}

	public Node selectRoot(){
		Integer root = getRoot();

		return selectNode(root);
	}

	public Node selectNode(Integer identifier){
		Map<Integer, Node> nodes = getNodes();

		Node node = nodes.get(identifier);
		if(node == null){
			throw new IllegalArgumentException();
		}

		return node;
	}

	public List<Node> selectNodes(Collection<Integer> identifiers){
		Map<Integer, Node> nodes = getNodes();

		List<Node> result = new ArrayList<>();

		for(Integer identifier : identifiers){
			Node node = nodes.get(identifier);

			if(node == null){
				throw new IllegalArgumentException();
			}

			result.add(node);
		}

		return result;
	}

	public String getIdentifier(){
		return getString("_identifier");
	}

	public Map<Integer, Node> getNodes(){
		return get("_nodes", Map.class);
	}

	public Integer getRoot(){
		return getInteger("root");
	}
}