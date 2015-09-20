/*
 * Copyright (c) 2015 Villu Ruusmann
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
package sklearn.tree;

import java.util.List;
import java.util.Map;

import com.google.common.primitives.Doubles;
import com.google.common.primitives.Ints;
import joblib.NDArrayWrapper;
import org.jpmml.sklearn.CClassDict;

public class Tree extends CClassDict {

	public Tree(String module, String name){
		super(module, name);
	}

	@Override
	public void __init__(Object[] args){
		super.__setstate__(createAttributeMap(INIT_ATTRIBUTES, args));
	}

	public int[] getChildrenLeft(){
		return Ints.toArray(getNodeAttribute("left_child"));
	}

	public int[] getChildrenRight(){
		return Ints.toArray(getNodeAttribute("right_child"));
	}

	public int[] getFeature(){
		return Ints.toArray(getNodeAttribute("feature"));
	}

	public double[] getThreshold(){
		return Doubles.toArray(getNodeAttribute("threshold"));
	}

	public int[] getNodeSamples(){
		return Ints.toArray(getNodeAttribute("n_node_samples"));
	}

	public double[] getWeightedNodeSamples(){
		return Doubles.toArray(getNodeAttribute("weighted_n_node_samples"));
	}

	public double[] getValues(){
		NDArrayWrapper values = (NDArrayWrapper)get("values");

		return Doubles.toArray((List<? extends Number>)values.getContent());
	}

	private List<? extends Number> getNodeAttribute(String key){
		NDArrayWrapper nodes = (NDArrayWrapper)get("nodes");

		Map<String, ?> content = (Map<String, ?>)nodes.getContent();

		return (List<? extends Number>)content.get(key);
	}

	private static final String[] INIT_ATTRIBUTES = {
		"n_features",
		"n_classes",
		"n_outputs"
	};
}