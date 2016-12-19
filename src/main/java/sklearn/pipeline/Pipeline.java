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

import org.dmg.pmml.Model;
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
	public int getNumberOfFeatures(){
		List<Transformer> transformers = getTransformers();
		Estimator estimator = getEstimator();

		if(transformers.size() > 0){

			for(Transformer transformer : transformers){
				HasNumberOfFeatures hasNumberOfFeatures = (HasNumberOfFeatures)transformer;

				return hasNumberOfFeatures.getNumberOfFeatures();
			}
		}

		return estimator.getNumberOfFeatures();
	}

	@Override
	public Schema createSchema(FeatureMapper featureMapper){
		List<Transformer> transformers = getTransformers();
		Estimator estimator = getEstimator();

		if(transformers.size() > 0 && featureMapper.isEmpty()){
			throw new IllegalArgumentException();
		}

		return estimator.createSchema(featureMapper);
	}

	@Override
	public Model encodeModel(Schema schema){
		List<Transformer> transformers = getTransformers();
		Estimator estimator = getEstimator();

		if(transformers.size() > 0){
			List<Feature> features = schema.getFeatures();

			for(Transformer transformer : transformers){
				HasNumberOfFeatures hasNumberOfFeatures = (HasNumberOfFeatures)transformer;

				if(features.size() != hasNumberOfFeatures.getNumberOfFeatures()){
					throw new IllegalArgumentException();
				}

				features = transformer.encodeFeatures(null, features, null);
			}

			schema = new Schema(schema.getTargetField(), schema.getTargetCategories(), schema.getActiveFields(), features);
		}

		return estimator.encodeModel(schema);
	}

	public List<Transformer> getTransformers(){
		List<Object[]> steps = getSteps();

		if(steps.size() < 1){
			throw new IllegalArgumentException();
		}

		List<Object[]> transformerSteps = steps.subList(0, steps.size() - 1);

		return TransformerUtil.asTransformerList(TupleUtil.extractElement(transformerSteps, 1));
	}

	public Estimator getEstimator(){
		List<Object[]> steps = getSteps();

		if(steps.size() < 1){
			throw new IllegalArgumentException();
		}

		Object[] estimatorStep = steps.get(steps.size() - 1);

		String name = (String)estimatorStep[0];
		Estimator estimator = EstimatorUtil.asEstimator(estimatorStep[1]);

		return estimator;
	}

	public List<Object[]> getSteps(){
		return (List)get("steps");
	}
}