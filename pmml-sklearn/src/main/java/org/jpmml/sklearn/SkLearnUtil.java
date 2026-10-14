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
import sklearn.Step;
import sklearn.StepCastFunction;
import sklearn.StepUtil;
import sklearn.pipeline.SkLearnPipeline;

public class SkLearnUtil {

	private SkLearnUtil(){
	}

	static
	public void initOnce(){

		if(!SkLearnUtil.initialized){
			init();

			SkLearnUtil.initialized = true;
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

	static
	public PMML encodePMML(PMMLEncodable encodable, Map<String, ?> options){

		try {
			if(encodable instanceof Step){
				return StepUtil.encodePMML((Step & PMMLEncodable)encodable, options);
			} else

			{
				return encodable.encodePMML(options);
			}
		} catch(SkLearnException se){
			throw se.ensureContext(encodable);
		} catch(Exception e){
			throw new SkLearnException("Failed to convert the object (" + ClassDictUtil.formatClass(encodable) + ") to PMML", e)
				.setContext(encodable);
		}
	}

	static
	private void init(){
		@SuppressWarnings("unused")
		SkLearnEncoder encoder = new SkLearnEncoder();
	}

	private static boolean initialized = false;
}