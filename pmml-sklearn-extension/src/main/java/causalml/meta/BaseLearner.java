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
package causalml.meta;

import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

import org.dmg.pmml.Model;
import org.jpmml.converter.Schema;
import org.jpmml.python.ClassDictUtil;
import sklearn.Estimator;
import sklearn.EstimatorCastFunction;
import sklearn.Regressor;

abstract
public class BaseLearner<E extends Estimator> extends Regressor {

	public BaseLearner(String module, String name){
		super(module, name);
	}

	abstract
	public Class<? extends E> getEstimatorClass();

	abstract
	public Model encodeEstimator(Role role, E estimator, Schema schema);

	public String getControlName(){
		return getString("control_name");
	}

	public Map<String, E> getModels(String name){
		Class<? extends E> estimatorClazz = getEstimatorClass();

		return getModels(name, estimatorClazz);
	}

	public <E extends Estimator> Map<String, E> getModels(String name, Class<? extends E> estimatorClazz){
		Map<String, ?> models = getDict(name);

		Function<Object, E> valueFunction = new EstimatorCastFunction<E>(estimatorClazz){

			@Override
			protected String formatMessage(Object object){
				return "The model object (" + ClassDictUtil.formatClass(object) + ") is not a supported Estimator";
			}
		};

		Map<String, E> result = (models.entrySet()).stream()
			.collect(Collectors.toMap(entry -> entry.getKey(), entry -> valueFunction.apply(entry.getValue())));

		return result;
	}

	public List<String> getTreatmentGroups(){
		return getStringArray("t_groups");
	}
}