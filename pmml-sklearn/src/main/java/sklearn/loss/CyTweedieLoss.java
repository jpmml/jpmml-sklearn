/*
 * Copyright (c) 2025 Villu Ruusmann
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
package sklearn.loss;

public class CyTweedieLoss extends CyLossFunction {

	public CyTweedieLoss(String module, String name){
		super(module, name);
	}

	@Override
	public void __init__(Object[] args){
		super.__setstate__(INIT_ATTRIBUTES, args);
	}

	public Number getPower(){
		return getNumber("power");
	}

	private static final String[] INIT_ATTRIBUTES = {
		"power"
	};
}