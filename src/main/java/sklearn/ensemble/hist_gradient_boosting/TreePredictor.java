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
package sklearn.ensemble.hist_gradient_boosting;

import java.util.List;

import com.google.common.primitives.Doubles;
import com.google.common.primitives.Ints;
import org.jpmml.python.PythonObject;

public class TreePredictor extends PythonObject {

	public TreePredictor(String module, String name){
		super(module, name);
	}

	public double[] getValues(){
		return Doubles.toArray(getNodeAttribute("value"));
	}

	public int[] getCount(){
		return Ints.toArray(getNodeAttribute("count"));
	}

	public int[] getFeatureIdx(){
		return Ints.toArray(getNodeAttribute("feature_idx"));
	}

	public double[] getThreshold(){
		return Doubles.toArray(getNodeAttribute("threshold"));
	}

	public int[] getMissingGoToLeft(){
		return Ints.toArray(getNodeAttribute("missing_go_to_left"));
	}

	public int[] getLeft(){
		return Ints.toArray(getNodeAttribute("left"));
	}

	public int[] getRight(){
		return Ints.toArray(getNodeAttribute("right"));
	}

	public int[] isLeaf(){
		return Ints.toArray(getNodeAttribute("is_leaf"));
	}

	public int[] getBinThreshhold(){
		return Ints.toArray(getNodeAttribute("bin_threshold"));
	}

	private List<? extends Number> getNodeAttribute(String key){
		return (List)getArray("nodes", key);
	}
}