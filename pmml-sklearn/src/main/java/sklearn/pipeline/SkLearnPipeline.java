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

import java.util.ArrayList;
import java.util.List;

import com.google.common.collect.Lists;
import org.dmg.pmml.Model;
import org.dmg.pmml.PMML;
import org.jpmml.converter.Feature;
import org.jpmml.converter.Schema;
import org.jpmml.python.CastFunction;
import org.jpmml.python.CastUtil;
import org.jpmml.python.ClassDictUtil;
import org.jpmml.python.TupleUtil;
import org.jpmml.sklearn.Encodable;
import org.jpmml.sklearn.EncodableUtil;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn.Composite;
import sklearn.Estimator;
import sklearn.EstimatorUtil;
import sklearn.Initializer;
import sklearn.PassThrough;
import sklearn.SkLearnSteps;
import sklearn.Step;
import sklearn.Transformer;
import sklearn.TransformerUtil;

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
				return "The transformer object (" + ClassDictUtil.formatClass(object) + ") is not a supported Transformer";
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
				return "The transformer object of the final step (" + ClassDictUtil.formatClass(object) + ") is not a supported Estimator";
			}
		};

		return castFunction.apply(estimator);
	}

	@Override
	public Transformer getHead(){

		if(hasTransformers()){
			List<? extends Transformer> transformers = getTransformers();

			Transformer transformer = transformers.get(0);

			return TransformerUtil.getHead(transformer);
		} // End if

		if(hasFinalEstimator()){
			Estimator estimator = getFinalEstimator();

			return EstimatorUtil.getHead(estimator);
		}

		return null;
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

	protected List<String> initLabel(Estimator estimator, List<String> targetFields, SkLearnEncoder encoder){

		if(estimator != null && estimator.isSupervised()){

			if(targetFields == null){
				targetFields = initTargetFields(estimator);
			}

			encoder.initLabel(estimator, targetFields);
		}

		return targetFields;
	}

	protected List<String> initTargetFields(Estimator estimator){
		return EncodableUtil.generateOutputNames(estimator);
	}

	protected List<String> initFeatures(Estimator estimator, List<String> activeFields, SkLearnEncoder encoder){
		Step featureInitializer = estimator;

		try {
			Transformer transformer = getHead();

			if(transformer != null){
				featureInitializer = transformer;

				if(!(transformer instanceof Initializer)){

					if(activeFields == null){
						activeFields = initActiveFields(transformer);
					}

					encoder.initFeatures(transformer, activeFields);
				}

				// XXX
				List<Feature> features = new ArrayList<>();
				features.addAll(encoder.getFeatures());

				features = super.encodeFeatures(features, encoder);

				encoder.setFeatures(features);
			} else

			if(estimator != null){

				if(activeFields == null){
					activeFields = initActiveFields(estimator);
				}

				encoder.initFeatures(estimator, activeFields);
			}
		} catch(UnsupportedOperationException uoe){
			throw new IllegalArgumentException("The transformer object of the first step (" + ClassDictUtil.formatClass(featureInitializer) + ") does not specify feature type information", uoe);
		}

		return activeFields;
	}

	protected List<String> initActiveFields(Step step){
		return EncodableUtil.getOrGenerateFeatureNames(step);
	}

	public List<Object[]> getSteps(){
		return getTupleList("steps");
	}

	protected SkLearnPipeline setSteps(List<Object[]> steps){
		put("steps", steps);

		return this;
	}
}