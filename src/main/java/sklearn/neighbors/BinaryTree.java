/*
 * Copyright (c) 2016 Villu Ruusmann
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
package sklearn.neighbors;

import org.jpmml.sklearn.CClassDict;

public class BinaryTree extends CClassDict {

	public BinaryTree(String module, String name){
		super(module, name);
	}

	@Override
	public void __init__(Object[] args){
		super.__setstate__(createAttributeMap(INIT_ATTRIBUTES, args));
	}

	@Override
	public void __setstate__(Object[] args){
		super.__setstate__(createAttributeMap(SETSTATE_ATTRIBUTES, args));
	}

	private static final String[] INIT_ATTRIBUTES = {
		"data"
	};

	private static final String[] SETSTATE_ATTRIBUTES = {
		"data_arr",
		"idx_array_arr",
		"node_data_arr",
		"node_bounds_arr",
		"leaf_size",
		"n_levels",
		"n_nodes",
		"n_trims",
		"n_leaves",
		"n_splits",
		"n_calls",
		"dist_metric"
	};
}