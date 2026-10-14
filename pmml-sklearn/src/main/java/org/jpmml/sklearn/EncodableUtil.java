/*
 * Copyright (c) 2023 Villu Ruusmann
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
package org.jpmml.sklearn;

import java.util.Map;

import org.dmg.pmml.PMML;
import org.jpmml.converter.PMMLEncodable;
import org.jpmml.python.CastFunction;
import org.jpmml.python.ClassDictUtil;
import sklearn.Estimator;
import sklearn.EstimatorUtil;
import sklearn.Step;
import sklearn.StepCastFunction;
import sklearn.Transformer;
import sklearn.TransformerUtil;
import sklearn.pipeline.SkLearnPipeline;
import sklearn2pmml.HasPMMLOptions;
import sklearn2pmml.pipeline.PMMLPipeline;

public class EncodableUtil {

	private EncodableUtil(){
	}

	/**
	 * @see EstimatorUtil#encodePMML(Estimator)
	 * @see TransformerUtil#encodePMML(Transformer)
	 */
	static
	public PMML encodePMML(PMMLEncodable encodable){

		try {
			return encodable.encodePMML();
		} catch(SkLearnException se){
			throw se.ensureContext(encodable);
		} catch(Exception e){
			throw new SkLearnException("Failed to convert the object (" + ClassDictUtil.formatClass(encodable) + ") to PMML", e)
				.setContext(encodable);
		}
	}

	static
	public void configure(PMMLEncodable encodable, Map<String, ?> pmmlOptions){

		if(encodable instanceof PMMLPipeline){
			PMMLPipeline pipeline = (PMMLPipeline)encodable;

			pipeline.configure(pmmlOptions);
		} else

		{
			HasPMMLOptions<?> hasPMMLOptions = (HasPMMLOptions<?>)encodable;

			hasPMMLOptions.setPMMLOptions(pmmlOptions);
		}
	}

	static
	public PMMLEncodable toEncodable(Object object){

		if(object instanceof PMMLEncodable){
			PMMLEncodable encodable = (PMMLEncodable)object;

			return encodable;
		}

		CastFunction<Step> castFunction = new StepCastFunction<>(Step.class);

		Step step = castFunction.apply(object);

		// A Castable non-Step class may turn into a Step subclass during "deep casting"
		if(step instanceof PMMLEncodable){
			PMMLEncodable encodable = (PMMLEncodable)step;

			return encodable;
		}

		SkLearnPipeline pipeline = new SkLearnPipeline(){

			@Override
			// No-op, to leave an identification mark into exception stack traces
			public PMML encodePMML(Map<String, ?> options){
				return super.encodePMML(options);
			}
		}
			.setOnlyStep("estimator", step);

		return pipeline;
	}
}