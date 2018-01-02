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
package sklearn;

import java.util.List;

import sklearn.pipeline.FeatureUnion;
import sklearn.pipeline.Pipeline;

public class TransformerUtil {

	private TransformerUtil(){
	}

	static
	public Transformer getHead(List<? extends Transformer> transformers){

		while(transformers.size() > 0){
			Transformer transformer = transformers.get(0);

			if(transformer instanceof FeatureUnion){
				FeatureUnion featureUnion = (FeatureUnion)transformer;

				transformers = featureUnion.getTransformers();
			} else

			if(transformer instanceof Pipeline){
				Pipeline pipeline = (Pipeline)transformer;

				transformers = pipeline.getTransformers();
			} else

			{
				return transformer;
			}
		}

		return null;
	}
}