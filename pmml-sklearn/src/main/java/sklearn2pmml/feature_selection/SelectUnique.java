/*
 * Copyright (c) 2019 Villu Ruusmann
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
package sklearn2pmml.feature_selection;

import java.util.List;

import sklearn.Selector;

public class SelectUnique extends Selector {

	public SelectUnique(String module, String name){
		super(module, name);
	}

	@Override
	public int getNumberOfFeatures(){
		int[] shape = getSupportMaskShape();

		return shape[0];
	}

	@Override
	public List<Boolean> getSupportMask(){
		return getBooleanArray("support_mask_");
	}

	public int[] getSupportMaskShape(){
		return getArrayShape("support_mask_", 1);
	}
}