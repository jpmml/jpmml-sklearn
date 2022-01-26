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

import java.util.Arrays;
import java.util.List;

import com.google.common.primitives.Doubles;
import com.google.common.primitives.Ints;
import org.jpmml.python.PythonObject;

public class TreePredictor extends PythonObject {

	public TreePredictor(String module, String name){
		super(module, name);
	}

	public int[] getRawLeftCatBitsets(){

		// SkLearn 0.23
		if(!containsKey("raw_left_cat_bitsets")){
			return null;
		}

		return Ints.toArray(getIntegerArray("raw_left_cat_bitsets"));
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
		List<? extends Number> threshold = getNodeAttribute("threshold");

		// SkLearn 0.23
		if(threshold != null){
			return Doubles.toArray(threshold);
		}

		// SkLearn 0.24+
		return Doubles.toArray(getNodeAttribute("num_threshold"));
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

	public int[] isCategorical(){
		List<? extends Number> isCategorical = getNodeAttribute("is_categorical");

		// SkLearn 0.23
		if(isCategorical == null){
			return null;
		}

		// SkLearn 0.24+
		return Ints.toArray(isCategorical);
	}

	public int[] getBitsetIdx(){
		List<? extends Number> bitsetIdx = getNodeAttribute("bitset_idx");

		// SkLearn 0.23
		if(bitsetIdx == null){
			return null;
		}

		// SkLearn 0.24+
		return Ints.toArray(bitsetIdx);
	}

	private List<? extends Number> getNodeAttribute(String key){
		return (List)getArray("nodes", key);
	}

	public static final List<String> DTYPE_PREDICTOR_OLD = Arrays.asList("value", "count", "feature_idx", "threshold", "missing_go_to_left", "left", "right", "gain", "depth", "is_leaf", "bin_threshold");
	public static final List<String> DTYPE_PREDICTOR_NEW = Arrays.asList("value", "count", "feature_idx", "num_threshold", "missing_go_to_left", "left", "right", "gain", "depth", "is_leaf", "bin_threshold", "is_categorical", "bitset_idx");
}