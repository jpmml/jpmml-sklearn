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

import java.util.Collections;
import java.util.List;

import org.jpmml.python.HasArray;
import org.jpmml.python.PythonObject;
import scipy.sparse.CSRMatrix;

public class TfidfTransformer extends PythonObject {

	public TfidfTransformer(String module, String name){
		super(module, name);
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
		return getOptionalEnum("norm", this::getOptionalString, Collections.emptyList());
	}

	public Boolean getSublinearTf(){
		return getBoolean("sublinear_tf");
	}

	public Boolean getUseIdf(){
		return getBoolean("use_idf");
	}
}