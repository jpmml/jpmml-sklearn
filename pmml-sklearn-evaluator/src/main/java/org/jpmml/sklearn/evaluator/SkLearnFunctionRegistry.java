/*
 * Copyright (c) 2025 Villu Ruusmann
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
package org.jpmml.sklearn.evaluator;

import java.util.Collections;
import java.util.Map;
import java.util.Objects;
import java.util.function.Predicate;

import org.jpmml.evaluator.Function;
import org.jpmml.evaluator.FunctionRegistry;

/**
 * @see FunctionRegistry
 */
public class SkLearnFunctionRegistry {

	private SkLearnFunctionRegistry(){
	}

	static
	public void publish(String name){
		publish(key -> Objects.equals(name, key));
	}

	static
	public void publishAll(){
		publish(key -> true);
	}

	static
	private void publish(Predicate<String> predicate){
		(SkLearnFunctionRegistry.rexpFunctions.entrySet()).stream()
			.filter(entry -> predicate.test(entry.getKey()))
			.forEach(entry -> FunctionRegistry.putFunction(entry.getKey(), entry.getValue()));

		(SkLearnFunctionRegistry.rexpFunctionClazzes.entrySet()).stream()
			.filter(entry -> predicate.test(entry.getKey()))
			.forEach(entry -> FunctionRegistry.putFunction(entry.getKey(), entry.getValue()));
	}

	private static final Map<String, Function> rexpFunctions = Collections.emptyMap();
	private static final Map<String, Class<? extends Function>> rexpFunctionClazzes = Collections.emptyMap();
}