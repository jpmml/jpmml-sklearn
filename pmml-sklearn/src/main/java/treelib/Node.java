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

import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import org.jpmml.python.PythonObject;

public class Node extends PythonObject {

	public Node(String module, String name){
		super(module, name);
	}

	public Node selectPredecessor(Tree tree){
		Map<String, ?> predecessor = getPredecessor();

		Integer identifier = (Integer)predecessor.get(tree.getIdentifier());

		return tree.selectNode(identifier);
	}

	public List<Node> selectSuccessors(Tree tree){
		Map<String, ?> successors = getSuccessors();

		Collection<Integer> identifiers = (Collection<Integer>)successors.get(tree.getIdentifier());
		if(identifiers == null){
			return Collections.emptyList();
		}

		return tree.selectNodes(identifiers);
	}

	public Boolean getExpanded(){
		return getBoolean("expanded");
	}

	public Integer getIdentifier(){
		return getInteger("_identifier");
	}

	public Map<String, ?> getPredecessor(){
		return getDict("_predecessor");
	}

	public Map<String, ?> getSuccessors(){
		return getDict("_successors");
	}

	public Object getTag(){
		return getTag(Object.class);
	}

	public <E> E getTag(Class<? extends E> clazz){
		return get("_tag", clazz);
	}
}