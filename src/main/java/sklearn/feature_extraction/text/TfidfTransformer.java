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

import java.util.List;

import org.jpmml.sklearn.PyClassDict;
import scipy.sparse.CSRMatrix;

public class TfidfTransformer extends PyClassDict {

	public TfidfTransformer(String module, String name){
		super(module, name);
	}

	public Number getWeight(int index){
		CSRMatrix idfDiag = get("_idf_diag", CSRMatrix.class);

		List<?> data = idfDiag.getData();

		return (Number)data.get(index);
	}

	public String getNorm(){
		return getOptionalString("norm");
	}

	public Boolean getSublinearTf(){
		return getBoolean("sublinear_tf");
	}

	public Boolean getUseIdf(){
		return getBoolean("use_idf");
	}
}