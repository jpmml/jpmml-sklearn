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

import com.google.common.collect.Lists;
import org.dmg.pmml.Model;
import org.dmg.pmml.PMML;
import org.jpmml.converter.Schema;
import org.jpmml.python.CastFunction;
import org.jpmml.python.CastUtil;
import org.jpmml.python.ClassDictUtil;
import org.jpmml.python.TupleUtil;
import org.jpmml.sklearn.Encodable;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn.Composite;
import sklearn.Estimator;
import sklearn.PassThrough;
import sklearn.SkLearnSteps;
import sklearn.Step;
import sklearn.StepUtil;
import sklearn.Transformer;

public class SkLearnPipeline extends Composite implements Encodable {

	public SkLearnPipeline(){
		this("sklearn.pipeline", "Pipeline");
	}

	public SkLearnPipeline(String module, String name){
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
				return "The object (" + ClassDictUtil.formatClass(object) + ") is not a supported Transformer";
			}
		};

		return Lists.transform(transformers, castFunction);
	}

	@Override
	public Estimator getFinalEstimator(){
		return getFinalEstimator(Estimator.class);
	}

	@Override
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
				return "The object (" + ClassDictUtil.formatClass(object) + ") is not a supported Estimator";
			}
		};

		return castFunction.apply(estimator);
	}

	@Override
	public Step getHead(){
		List<Object[]> steps = getSteps();

		if(steps.isEmpty()){
			throw new IllegalArgumentException("Expected one or more steps, got zero steps");
		}

		Object[] headStep = steps.get(0);

		Object step = TupleUtil.extractElement(headStep, 1);

		CastFunction<Step> castFunction = new CastFunction<Step>(Step.class){

			@Override
			public Step apply(Object object){

				if((SkLearnSteps.PASSTHROUGH).equals(object)){
					return null;
				}

				return super.apply(object);
			}

			@Override
			public String formatMessage(Object object){
				return "The object (" + ClassDictUtil.formatClass(object) + ") is not a supported Transformer or Estimator";
			}
		};

		step = castFunction.apply(step);

		return StepUtil.getHead((Step)step);
	}

	@Override
	public PMML encodePMML(){
		SkLearnEncoder encoder = new SkLearnEncoder();

		Estimator estimator = null;

		if(hasFinalEstimator()){
			estimator = getFinalEstimator();
		}

		initLabel(estimator, null, encoder);
		initFeatures(estimator, null, encoder);

		if(estimator == null){
			return encoder.encodePMML(null);
		}

		Schema schema = encoder.createSchema();

		Model model = estimator.encode(schema);

		encoder.setModel(model);

		return encoder.encodePMML(model);
	}

	public List<Object[]> getSteps(){
		return getTupleList("steps");
	}

	protected SkLearnPipeline setSteps(List<Object[]> steps){
		put("steps", steps);

		return this;
	}
}