/*
 * Copyright (c) 2022 Villu Ruusmann
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
package sklearn.pipeline;

import java.util.List;

import sklearn.Estimator;
import sklearn.HasHead;
import sklearn.Transformer;

public class PipelineUtil {

	private PipelineUtil(){
	}

	static
	public Transformer getHead(List<? extends Transformer> transformers, Estimator estimator){

		if(!transformers.isEmpty()){
			Transformer transformer = transformers.get(0);

			if(transformer instanceof HasHead){
				HasHead hasHead = (HasHead)transformer;

				return hasHead.getHead();
			}

			return transformer;
		} // End if

		if(estimator != null){

			if(estimator instanceof HasHead){
				HasHead hasHead = (HasHead)estimator;

				return hasHead.getHead();
			}

			return null;
		}

		return null;
	}
}