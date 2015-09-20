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
package numpy;

import org.jpmml.sklearn.CClassDict;

public class DType extends CClassDict {

	private String descr = null;


	public DType(String module, String name){
		super(module, name);
	}

	@Override
	public void __init__(Object[] args){
		super.__setstate__(createAttributeMap(INIT_ATTRIBUTES, args));

		this.descr = (String)get("obj");
	}

	/**
	 * https://github.com/numpy/numpy/blob/master/numpy/core/src/multiarray/descriptor.c
	 */
	@Override
	public void __setstate__(Object[] args){
		super.__setstate__(createAttributeMap(SETSTATE_ATTRIBUTES, args));
	}

	public String getDescr(){

		if(this.descr == null){
			throw new IllegalStateException();
		}

		String order = (String)get("order");

		return (order != null ? (order + this.descr) : this.descr);
	}

	private static final String[] INIT_ATTRIBUTES = {
		"obj",
		"align",
		"copy"
	};

	private static final String[] SETSTATE_ATTRIBUTES = {
		"version",
		"order",
		"subdescr",
		"names",
		"values",
		"w_size",
		"alignment",
		"flags"
	};
}