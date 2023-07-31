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
package sklearn2pmml.pipeline;

import java.util.Collections;

import org.jpmml.python.CastFunction;
import org.jpmml.python.ClassDictUtil;
import sklearn.Step;

public class PMMLPipelineUtil {

	private PMMLPipelineUtil(){
	}

	static
	public PMMLPipeline toPMMLPipeline(Object object){

		if(object instanceof PMMLPipeline){
			PMMLPipeline pipeline = (PMMLPipeline)object;

			return pipeline;
		}

		CastFunction<Step> castFunction = new CastFunction<Step>(Step.class){

			@Override
			protected String formatMessage(Object object){
				return "The object (" + ClassDictUtil.formatClass(object) + ") is not a supported Transformer or Estimator";
			}
		};

		Step step = castFunction.apply(object);

		PMMLPipeline pipeline = new PMMLPipeline()
			.setSteps(Collections.singletonList(new Object[]{"estimator", step}));

		return pipeline;
	}
}