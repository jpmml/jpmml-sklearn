/*
 * Copyright (c) 2015 Villu Ruusmann
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
import org.jpmml.converter.CategoricalLabel;
import org.jpmml.sklearn.ClassDictUtil;

public class EstimatorUtil {

	private EstimatorUtil(){
	}

	static
	public List<?> getClasses(Estimator estimator){
		HasClasses hasClasses = (HasClasses)estimator;

		return hasClasses.getClasses();
	}

	static
	public Estimator asEstimator(Object object){
		return EstimatorUtil.estimatorFunction.apply(object);
	}

	static
	public List<Estimator> asEstimatorList(List<?> objects){
		return Lists.transform(objects, EstimatorUtil.estimatorFunction);
	}

	static
	public Classifier asClassifier(Object object){
		return EstimatorUtil.classifierFunction.apply(object);
	}

	static
	public List<? extends Classifier> asClassifierList(List<?> objects){
		return Lists.transform(objects, EstimatorUtil.classifierFunction);
	}

	static
	public Regressor asRegressor(Object object){
		return EstimatorUtil.regressorFunction.apply(object);
	}

	static
	public List<? extends Regressor> asRegressorList(List<?> objects){
		return Lists.transform(objects, EstimatorUtil.regressorFunction);
	}

	static
	public void checkSize(int size, CategoricalLabel categoricalLabel){

		if(categoricalLabel.size() != size){
			throw new IllegalArgumentException("Expected " + size + " class(es), got " + categoricalLabel.size() + " class(es)");
		}
	}

	private static final Function<Object, Estimator> estimatorFunction = new Function<Object, Estimator>(){

		@Override
		public Estimator apply(Object object){

			try {
				if(object == null){
					throw new NullPointerException();
				}

				return (Estimator)object;
			} catch(RuntimeException re){
				throw new IllegalArgumentException("The estimator object (" + ClassDictUtil.formatClass(object) + ") is not an Estimator or is not a supported Estimator subclass", re);
			}
		}
	};

	private static final Function<Object, Classifier> classifierFunction = new Function<Object, Classifier>(){

		@Override
		public Classifier apply(Object object){

			try {
				if(object == null){
					throw new NullPointerException();
				}

				return (Classifier)object;
			} catch(RuntimeException re){
				throw new IllegalArgumentException("The estimator object (" + ClassDictUtil.formatClass(object) + ") is not a Classifier or is not a supported Classifier subclass", re);
			}
		}
	};

	private static final Function<Object, Regressor> regressorFunction = new Function<Object, Regressor>(){

		@Override
		public Regressor apply(Object object){

			try {
				if(object == null){
					throw new NullPointerException();
				}

				return (Regressor)object;
			} catch(RuntimeException re){
				throw new IllegalArgumentException("The estimator object (" + ClassDictUtil.formatClass(object) + ") is not a Regressor or is not a supported Regressor subclass", re);
			}
		}
	};
}
