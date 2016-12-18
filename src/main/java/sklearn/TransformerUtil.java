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
}