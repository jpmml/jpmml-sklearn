/*
 * Copyright (c) 2026 Villu Ruusmann
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

import java.util.Collection;

import org.jpmml.converter.ExceptionUtil;
import org.jpmml.python.ClassDictUtil;
import org.jpmml.sklearn.SkLearnException;

public class TransformerCastException extends SkLearnException {

	public TransformerCastException(Transformer transformer, Collection<Class<?>> clazzes){
		super(formatMessage(transformer, clazzes));
	}

	public TransformerCastException(Transformer transformer, Collection<Class<?>> clazzes, Throwable cause){
		super(formatMessage(transformer, clazzes), cause);
	}

	static
	private String formatMessage(Transformer transformer, Collection<Class<?>> clazzes){
		return "The transformer object (" + ClassDictUtil.formatClass(transformer) + ") is not an instance of " + ExceptionUtil.formatClasses(clazzes);
	}
}