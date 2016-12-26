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
import java.util.Set;

import org.dmg.pmml.DataType;
import org.dmg.pmml.DefineFunction;
import org.dmg.pmml.Model;
import org.dmg.pmml.OpType;
import org.jpmml.converter.Feature;
import org.jpmml.converter.Schema;
import org.jpmml.sklearn.FeatureMapper;
import org.jpmml.sklearn.TupleUtil;
import sklearn.Estimator;
import sklearn.EstimatorUtil;
import sklearn.HasNumberOfFeatures;
import sklearn.Transformer;
import sklearn.TransformerUtil;

public class Pipeline extends Estimator {

	public Pipeline(String module, String name){
		super(module, name);
	}

	@Override
	public boolean isSupervised(){
		Estimator estimator = getEstimator();

		return estimator.isSupervised();
	}

	@Override
	public int getNumberOfFeatures(){
		List<Transformer> transformers = getTransformers();
		Estimator estimator = getEstimator();

		for(Transformer transformer : transformers){

			if(transformer instanceof HasNumberOfFeatures){
				HasNumberOfFeatures hasNumberOfFeatures = (HasNumberOfFeatures)transformer;

				return hasNumberOfFeatures.getNumberOfFeatures();
			}
		}

		return estimator.getNumberOfFeatures();
	}

	@Override
	public boolean requiresContinuousInput(){
		Estimator estimator = getEstimator();

		return estimator.requiresContinuousInput();
	}

	@Override
	public OpType getOpType(){
		List<Transformer> transformers = getTransformers();
		Estimator estimator = getEstimator();

		for(Transformer transformer : transformers){
			OpType opType = transformer.getOpType();

			if(opType != null){
				return opType;
			}
		}

		return estimator.getOpType();
	}

	@Override
	public DataType getDataType(){
		List<Transformer> transformers = getTransformers();
		Estimator estimator = getEstimator();

		for(Transformer transformer : transformers){
			DataType dataType = transformer.getDataType();

			if(dataType != null){
				return dataType;
			}
		}

		return estimator.getDataType();
	}

	@Override
	public Set<DefineFunction> encodeDefineFunctions(){
		Estimator estimator = getEstimator();

		return estimator.encodeDefineFunctions();
	}

	@Override
	public Model encodeModel(Schema schema){
		throw new UnsupportedOperationException();
	}

	@Override
	public Model encodeModel(Schema schema, FeatureMapper featureMapper){
		List<Transformer> transformers = getTransformers();
		Estimator estimator = getEstimator();

		if(transformers.size() > 0){
			List<String> ids = featureMapper.getIds();
			List<Feature> features = featureMapper.getFeatures();

			for(Transformer transformer : transformers){

				if(transformer instanceof HasNumberOfFeatures){
					HasNumberOfFeatures hasNumberOfFeatures = (HasNumberOfFeatures)transformer;

					int numberOfFeatures = hasNumberOfFeatures.getNumberOfFeatures();
					if(ids.size() != numberOfFeatures || features.size() != numberOfFeatures){
						throw new IllegalArgumentException();
					}
				}

				for(Feature feature : features){
					featureMapper.updateType(feature.getName(), transformer.getOpType(), transformer.getDataType());
				}

				features = transformer.encodeFeatures(ids, features, featureMapper);
			}

			schema = new Schema(schema.getTargetField(), schema.getTargetCategories(), schema.getActiveFields(), features);
		}

		List<Feature> features = schema.getFeatures();

		int numberOfFeatures = estimator.getNumberOfFeatures();
		if(features.size() != numberOfFeatures){
			throw new IllegalArgumentException();
		}

		return estimator.encodeModel(schema, featureMapper);
	}

	public List<Transformer> getTransformers(){
		List<Object[]> transformerSteps = getTransformerSteps();

		return TransformerUtil.asTransformerList(TupleUtil.extractElementList(transformerSteps, 1));
	}

	public List<Object[]> getTransformerSteps(){
		List<Object[]> steps = getSteps();

		if(steps.size() < 1){
			throw new IllegalArgumentException();
		}

		return steps.subList(0, steps.size() - 1);
	}

	public Estimator getEstimator(){
		Object[] estimatorStep = getEstimatorStep();

		return EstimatorUtil.asEstimator(TupleUtil.extractElement(estimatorStep, 1));
	}

	protected Object[] getEstimatorStep(){
		List<Object[]> steps = getSteps();

		if(steps.size() < 1){
			throw new IllegalArgumentException();
		}

		return steps.get(steps.size() - 1);
	}

	public List<Object[]> getSteps(){
		return (List)get("steps");
	}
}