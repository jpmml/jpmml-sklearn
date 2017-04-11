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

import com.google.common.base.Function;
import com.google.common.collect.Lists;
import org.jpmml.sklearn.ClassDictUtil;
import sklearn.pipeline.FeatureUnion;
import sklearn.pipeline.Pipeline;

public class TransformerUtil {

	private TransformerUtil(){
	}

	static
	public Transformer asTransformer(Object object){
		return TransformerUtil.transformerFunction.apply(object);
	}

	static
	public List<Transformer> asTransformerList(List<?> objects){
		return Lists.transform(objects, TransformerUtil.transformerFunction);
	}

	static
	public Selector asSelector(Object object){
		return TransformerUtil.selectorFunction.apply(object);
	}

	static
	public List<Selector> asSelectorList(List<?> objects){
		return Lists.transform(objects, TransformerUtil.selectorFunction);
	}

	static
	public Transformer getHead(List<Transformer> transformers){

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

	private static final Function<Object, Transformer> transformerFunction = new Function<Object, Transformer>(){

		@Override
		public Transformer apply(Object object){

			try {
				if(object == null){
					throw new NullPointerException();
				}

				return (Transformer)object;
			} catch(RuntimeException re){
				throw new IllegalArgumentException("The transformer object (" + ClassDictUtil.formatClass(object) + ") is not a Transformer or is not a supported Transformer subclass", re);
			}
		}
	};

	private static final Function<Object, Selector> selectorFunction = new Function<Object, Selector>(){

		@Override
		public Selector apply(Object object){

			try {
				if(object == null){
					throw new NullPointerException();
				}

				return (Selector)object;
			} catch(RuntimeException re){
				throw new IllegalArgumentException("The transformer object (" + ClassDictUtil.formatClass(object) + ") is not a Selector or is not a supported Selector subclass");
			}
		}
	};
}