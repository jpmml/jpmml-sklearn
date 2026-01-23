/*
 * Copyright (c) 2017 Villu Ruusmann
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
package sklearn.feature_extraction.text;

import java.util.Arrays;
import java.util.List;

import org.jpmml.converter.Feature;
import org.jpmml.python.HasArray;
import org.jpmml.sklearn.SkLearnEncoder;
import scipy.sparse.CSRMatrix;
import sklearn.SkLearnTransformer;

public class TfidfTransformer extends SkLearnTransformer {

	public TfidfTransformer(String module, String name){
		super(module, name);
	}

	@Override
	public List<Feature> encodeFeatures(List<Feature> features, SkLearnEncoder encoder){
		throw new UnsupportedOperationException();
	}

	public Number getWeight(int index){
		List<?> data;

		// SkLearn 1.4.2
		if(hasattr("_idf_diag")){
			CSRMatrix idfDiag = get("_idf_diag", CSRMatrix.class);

			data = idfDiag.getData();
		} else

		// SkLearn 1.5+
		{
			HasArray idf = getArray("idf_");

			data = idf.getArrayContent();
		}

		return (Number)data.get(index);
	}

	public String getNorm(){
		return getOptionalEnum("norm", this::getOptionalString, Arrays.asList(TfidfTransformer.NORM_L1, TfidfTransformer.NORM_L2));
	}

	public Boolean getSublinearTf(){
		return getBoolean("sublinear_tf");
	}

	public Boolean getUseIdf(){
		return getBoolean("use_idf");
	}

	static final String NORM_L1 = "l1";
	static final String NORM_L2 = "l2";
}