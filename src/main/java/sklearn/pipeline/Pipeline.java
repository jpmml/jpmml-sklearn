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

import org.dmg.pmml.DataType;
import org.dmg.pmml.OpType;
import org.jpmml.converter.Feature;
import org.jpmml.sklearn.ClassDictUtil;
import org.jpmml.sklearn.SkLearnEncoder;
import org.jpmml.sklearn.TupleUtil;
import sklearn.Estimator;
import sklearn.HasNumberOfFeatures;
import sklearn.Transformer;

public class Pipeline extends Transformer {

	public Pipeline(){
		this("sklearn.pipeline", "Pipeline");
	}

	public Pipeline(String module, String name){
		super(module, name);
	}

	@Override
	public OpType getOpType(){
		List<? extends Transformer> transformers = getTransformers();

		for(Transformer transformer : transformers){
			return transformer.getOpType();
		}

		throw new IllegalArgumentException();
	}

	@Override
	public DataType getDataType(){
		List<? extends Transformer> transformers = getTransformers();

		for(Transformer transformer : transformers){
			return transformer.getDataType();
		}

		throw new IllegalArgumentException();
	}

	@Override
	public List<Feature> encodeFeatures(List<Feature> features, SkLearnEncoder encoder){
		List<? extends Transformer> transformers = getTransformers();

		for(Transformer transformer : transformers){

			if(transformer instanceof HasNumberOfFeatures){
				HasNumberOfFeatures hasNumberOfFeatures = (HasNumberOfFeatures)transformer;

				int numberOfFeatures = hasNumberOfFeatures.getNumberOfFeatures();
				if(numberOfFeatures > -1){
					ClassDictUtil.checkSize(numberOfFeatures, features);
				}
			}

			features = transformer.updateAndEncodeFeatures(features, encoder);
		}

		return features;
	}

	public boolean hasFinalEstimator(){
		List<Object[]> steps = getSteps();

		if(steps.size() < 1){
			return false;
		}

		Object[] finalStep = steps.get(steps.size() - 1);

		Object estimator = TupleUtil.extractElement(finalStep, 1);

		return Estimator.class.isInstance(estimator);
	}

	public List<? extends Transformer> getTransformers(){
		List<Object[]> steps = getSteps();

		if(hasFinalEstimator()){
			steps = steps.subList(0, steps.size() - 1);
		}

		return TupleUtil.extractElementList(steps, 1, Transformer.class);
	}

	public Estimator getFinalEstimator(){
		List<Object[]> steps = getSteps();

		if(steps.size() < 1){
			throw new IllegalArgumentException("Expected one or more steps, got zero steps");
		}

		Object[] finalStep = steps.get(steps.size() - 1);

		try {
			return TupleUtil.extractElement(finalStep, 1, Estimator.class);
		} catch(IllegalArgumentException iae){
			Object estimator = TupleUtil.extractElement(finalStep, 1);

			throw new IllegalArgumentException("The transformer object of the final step (" + ClassDictUtil.formatClass(estimator) + ") is not a supported Estimator", iae);
		}
	}

	public List<Object[]> getSteps(){
		return getTupleList("steps");
	}
}