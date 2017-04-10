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
import sklearn.EstimatorUtil;
import sklearn.HasNumberOfFeatures;
import sklearn.Transformer;
import sklearn.TransformerUtil;
import sklearn_pandas.DataFrameMapper;

public class Pipeline extends Transformer implements HasNumberOfFeatures {

	public Pipeline(String module, String name){
		super(module, name);
	}

	@Override
	public OpType getOpType(){
		DataFrameMapper mapper = getMapper();
		List<Transformer> transformers = getTransformers();
		Estimator estimator = getEstimator();

		if(mapper != null){
			throw new IllegalArgumentException();
		}

		for(Transformer transformer : transformers){
			return transformer.getOpType();
		}

		if(estimator != null){
			return estimator.getOpType();
		}

		throw new IllegalArgumentException();
	}

	@Override
	public DataType getDataType(){
		DataFrameMapper mapper = getMapper();
		List<Transformer> transformers = getTransformers();
		Estimator estimator = getEstimator();

		if(mapper != null){
			throw new IllegalArgumentException();
		}

		for(Transformer transformer : transformers){
			return transformer.getDataType();
		}

		if(estimator != null){
			return estimator.getDataType();
		}

		throw new IllegalArgumentException();
	}

	@Override
	public int getNumberOfFeatures(){
		DataFrameMapper mapper = getMapper();
		List<Transformer> transformers = getTransformers();
		Estimator estimator = getEstimator();

		if(mapper != null){
			return -1;
		}

		for(Transformer transformer : transformers){

			if(transformer instanceof HasNumberOfFeatures){
				HasNumberOfFeatures hasNumberOfFeatures = (HasNumberOfFeatures)transformer;

				return hasNumberOfFeatures.getNumberOfFeatures();
			}

			return -1;
		}

		if(estimator != null){
			return estimator.getNumberOfFeatures();
		}

		throw new IllegalArgumentException();
	}

	@Override
	public List<Feature> encodeFeatures(List<String> ids, List<Feature> features, SkLearnEncoder encoder){
		DataFrameMapper mapper = getMapper();
		Estimator estimator = getEstimator();

		if(mapper != null || estimator != null){
			throw new IllegalArgumentException();
		}

		return applyTransformers(ids, features, encoder);
	}

	@Override
	public List<Feature> encodeFeatures(SkLearnEncoder encoder){
		DataFrameMapper mapper = getMapper();

		if(mapper != null){
			mapper.encodeFeatures(encoder);
		}

		List<String> ids = encoder.getIds();
		List<Feature> features = encoder.getFeatures();

		return applyTransformers(ids, features, encoder);
	}

	public List<Feature> applyTransformers(List<String> ids, List<Feature> features, SkLearnEncoder encoder){
		List<Transformer> transformers = getTransformers();

		for(Transformer transformer : transformers){

			if(transformer instanceof HasNumberOfFeatures){
				HasNumberOfFeatures hasNumberOfFeatures = (HasNumberOfFeatures)transformer;

				int numberOfFeatures = hasNumberOfFeatures.getNumberOfFeatures();
				if(numberOfFeatures > -1){
					ClassDictUtil.checkSize(numberOfFeatures, ids, features);
				}
			}

			encoder.updateFeatures(features, transformer.getOpType(), transformer.getDataType());

			features = transformer.encodeFeatures(ids, features, encoder);
		}

		return features;
	}

	public DataFrameMapper getMapper(){
		Object[] mapperStep = getMapperStep();

		if(mapperStep != null){
			return (DataFrameMapper)TupleUtil.extractElement(mapperStep, 1);
		}

		return null;
	}

	protected Object[] getMapperStep(){
		List<Object[]> steps = getSteps();

		if(steps.size() > 0){
			Object firstStep = TupleUtil.extractElement(steps.get(0), 1);

			if(firstStep instanceof DataFrameMapper){
				return steps.get(0);
			}
		}

		return null;
	}

	public List<Transformer> getTransformers(){
		List<Object[]> transformerSteps = getTransformerSteps();

		return TransformerUtil.asTransformerList(TupleUtil.extractElementList(transformerSteps, 1));
	}

	protected List<Object[]> getTransformerSteps(){
		List<Object[]> steps = getSteps();

		if(steps.size() > 0){
			Object firstStep = TupleUtil.extractElement(steps.get(0), 1);

			if(firstStep instanceof DataFrameMapper){
				steps = steps.subList(1, steps.size());
			}
		} // End if

		if(steps.size() > 0){
			Object lastStep = TupleUtil.extractElement(steps.get(steps.size() - 1), 1);

			if(lastStep instanceof Estimator){
				steps = steps.subList(0, steps.size() - 1);
			}
		}

		return steps;
	}

	public Estimator getEstimator(){
		Object[] estimatorStep = getEstimatorStep();

		if(estimatorStep != null){
			return EstimatorUtil.asEstimator(TupleUtil.extractElement(estimatorStep, 1));
		}

		return null;
	}

	protected Object[] getEstimatorStep(){
		List<Object[]> steps = getSteps();

		if(steps.size() > 0){
			Object lastStep = TupleUtil.extractElement(steps.get(steps.size() - 1), 1);

			if(lastStep instanceof Estimator){
				return steps.get(steps.size() - 1);
			}
		}

		return null;
	}

	public List<Object[]> getSteps(){
		return (List)get("steps");
	}
}