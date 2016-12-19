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
import sklearn.Selector;
import sklearn.TransformerUtil;

public class Pipeline extends Estimator {

	public Pipeline(String module, String name){
		super(module, name);
	}

	@Override
	public int getNumberOfFeatures(){
		List<Selector> selectors = getSelectors();
		Estimator estimator = getEstimator();

		if(selectors.size() > 0){

			for(Selector selector : selectors){
				return selector.getNumberOfFeatures();
			}
		}

		return estimator.getNumberOfFeatures();
	}

	@Override
	public Schema createSchema(FeatureMapper featureMapper){
		List<Selector> selectors = getSelectors();
		Estimator estimator = getEstimator();

		if(selectors.size() > 0 && featureMapper.isEmpty()){
			throw new IllegalArgumentException();
		}

		return estimator.createSchema(featureMapper);
	}

	@Override
	public Model encodeModel(Schema schema){
		List<Selector> selectors = getSelectors();
		Estimator estimator = getEstimator();

		if(selectors.size() > 0){
			List<Feature> features = schema.getFeatures();

			for(Selector selector : selectors){
				int numberOfFeatures = selector.getNumberOfFeatures();

				if(features.size() != numberOfFeatures){
					throw new IllegalArgumentException();
				}

				features = selector.selectFeatures(features);
			}

			schema = new Schema(schema.getTargetField(), schema.getTargetCategories(), schema.getActiveFields(), features);
		}

		return estimator.encodeModel(schema);
	}

	public List<Selector> getSelectors(){
		List<Object[]> steps = getSteps();

		if(steps.size() < 1){
			throw new IllegalArgumentException();
		}

		List<Object[]> selectorSteps = steps.subList(0, steps.size() - 1);

		return TransformerUtil.asSelectorList(TupleUtil.extractElement(selectorSteps, 1));
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