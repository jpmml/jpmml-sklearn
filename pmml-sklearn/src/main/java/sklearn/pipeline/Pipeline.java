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
package sklearn.pipeline;

import java.util.List;
import java.util.Map;

import com.google.common.collect.Lists;
import org.jpmml.python.CastFunction;
import org.jpmml.python.CastUtil;
import org.jpmml.python.Castable;
import org.jpmml.python.ClassDictUtil;
import org.jpmml.python.TupleUtil;
import sklearn.Classifier;
import sklearn.Clusterer;
import sklearn.Composite;
import sklearn.Estimator;
import sklearn.HasHead;
import sklearn.HasPMMLOptions;
import sklearn.PassThrough;
import sklearn.Regressor;
import sklearn.SkLearnSteps;
import sklearn.Transformer;

public class Pipeline extends Composite implements Castable, HasHead, HasPMMLOptions<Pipeline> {

	public Pipeline(){
		this("sklearn.pipeline", "Pipeline");
	}

	public Pipeline(String module, String name){
		super(module, name);
	}

	@Override
	public boolean hasTransformers(){
		List<Object[]> steps = getSteps();

		if(steps.isEmpty()){
			return false;
		} // End if

		if(steps.size() == 1){
			return !hasFinalEstimator();
		} else

		{
			return true;
		}
	}

	@Override
	public boolean hasFinalEstimator(){
		List<Object[]> steps = getSteps();

		if(steps.isEmpty()){
			return false;
		}

		Object[] finalStep = steps.get(steps.size() - 1);

		Object estimator = TupleUtil.extractElement(finalStep, 1);

		if((SkLearnSteps.PASSTHROUGH).equals(estimator)){
			return true;
		}

		estimator = CastUtil.deepCastTo(estimator, Estimator.class);

		return (Estimator.class).isInstance(estimator);
	}

	@Override
	public List<? extends Transformer> getTransformers(){
		List<Object[]> steps = getSteps();

		if(hasFinalEstimator()){
			steps = steps.subList(0, steps.size() - 1);
		}

		List<?> transformers = TupleUtil.extractElementList(steps, 1);

		CastFunction<Transformer> castFunction = new CastFunction<Transformer>(Transformer.class){

			@Override
			public Transformer apply(Object object){

				if((object == null) || (SkLearnSteps.PASSTHROUGH).equals(object)){
					return PassThrough.INSTANCE;
				}

				return super.apply(object);
			}

			@Override
			public String formatMessage(Object object){
				return "The transformer object (" + ClassDictUtil.formatClass(object) + ") is not a supported Transformer";
			}
		};

		return Lists.transform(transformers, castFunction);
	}

	@Override
	public Estimator getFinalEstimator(){
		return getFinalEstimator(Estimator.class);
	}

	public <E extends Estimator> E getFinalEstimator(Class<? extends E> clazz){
		List<Object[]> steps = getSteps();

		if(steps.isEmpty()){
			throw new IllegalArgumentException("Expected one or more steps, got zero steps");
		}

		Object[] finalStep = steps.get(steps.size() - 1);

		Object estimator = TupleUtil.extractElement(finalStep, 1);

		CastFunction<E> castFunction = new CastFunction<E>(clazz){

			@Override
			public E apply(Object object){

				if((SkLearnSteps.PASSTHROUGH).equals(object)){
					return null;
				}

				return super.apply(object);
			}

			@Override
			public String formatMessage(Object object){
				return "The transformer object of the final step (" + ClassDictUtil.formatClass(object) + ") is not a supported Estimator";
			}
		};

		return castFunction.apply(estimator);
	}

	@Override
	public Object castTo(Class<?> clazz){

		if((Transformer.class).equals(clazz)){
			return toTransformer();
		} else

		if((Estimator.class).equals(clazz)){
			return toEstimator();
		} else

		if((Classifier.class).equals(clazz)){
			return toClassifier();
		} else

		if((Regressor.class).equals(clazz)){
			return toRegressor();
		}

		return this;
	}

	@Override
	public Transformer getHead(){
		List<? extends Transformer> transformers = getTransformers();
		Estimator estimator = null;

		if(hasFinalEstimator()){
			estimator = getFinalEstimator();
		}

		return PipelineUtil.getHead(transformers, estimator);
	}

	@Override
	public Map<String, ?> getPMMLOptions(){

		if(hasFinalEstimator()){
			Estimator estimator = getFinalEstimator();

			return estimator.getPMMLOptions();
		}

		return null;
	}

	@Override
	public Pipeline setPMMLOptions(Map<String, ?> pmmlOptions){

		if(hasFinalEstimator()){
			Estimator estimator = getFinalEstimator();

			estimator.setPMMLOptions(pmmlOptions);
		}

		return this;
	}

	public Transformer toTransformer(){

		if(hasFinalEstimator()){
			Estimator estimator = getFinalEstimator();

			if(estimator != null){
				throw new IllegalArgumentException("The pipeline ends with an estimator object");
			}
		}

		return new PipelineTransformer(this);
	}

	public Estimator toEstimator(){
		Estimator estimator = getFinalEstimator();

		if(estimator instanceof Classifier){
			return toClassifier();
		} else

		if(estimator instanceof Regressor){
			return toRegressor();
		} else

		if(estimator instanceof Clusterer){
			return toClusterer();
		}

		throw new IllegalArgumentException();
	}

	public Classifier toClassifier(){
		return new PipelineClassifier(this);
	}

	public Regressor toRegressor(){
		return new PipelineRegressor(this);
	}

	public Clusterer toClusterer(){
		return new PipelineClusterer(this);
	}

	public List<Object[]> getSteps(){
		return getTupleList("steps");
	}

	protected Pipeline setSteps(List<Object[]> steps){
		put("steps", steps);

		return this;
	}
}