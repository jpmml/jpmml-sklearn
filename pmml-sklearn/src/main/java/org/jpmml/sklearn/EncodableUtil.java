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

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import net.razorvine.pickle.objects.ClassDict;
import org.dmg.pmml.PMML;
import org.jpmml.python.Attribute;
import org.jpmml.python.CastFunction;
import org.jpmml.python.MissingAttributeException;
import sklearn.Estimator;
import sklearn.EstimatorUtil;
import sklearn.HasNumberOfFeatures;
import sklearn.HasNumberOfOutputs;
import sklearn.SkLearnFields;
import sklearn.Step;
import sklearn.StepCastFunction;
import sklearn.Transformer;
import sklearn.TransformerUtil;
import sklearn.pipeline.SkLearnPipeline;

public class EncodableUtil {

	private EncodableUtil(){
	}

	/**
	 * @see EstimatorUtil#encodePMML(Estimator)
	 * @see TransformerUtil#encodePMML(Transformer)
	 */
	static
	public PMML encodePMML(Encodable encodable){
		return encodable.encodePMML();
	}

	static
	public Encodable toEncodable(Object object){

		if(object instanceof Encodable){
			Encodable encodable = (Encodable)object;

			return encodable;
		}

		CastFunction<Step> castFunction = new StepCastFunction<>(Step.class);

		Step step = castFunction.apply(object);

		// A Castable non-Step class may turn into a Step subclass during "deep casting"
		if(step instanceof Encodable){
			Encodable encodable = (Encodable)step;

			return encodable;
		}

		SkLearnPipeline pipeline = new SkLearnPipeline(){

			{
				List<Object[]> steps = Collections.singletonList(new Object[]{"estimator", step});

				setSteps(steps);
			}

			@Override
			// No-op, to leave an identification mark into exception stack traces
			public PMML encodePMML(){
				return super.encodePMML();
			}
		};

		return pipeline;
	}

	static
	public List<String> getOrGenerateFeatureNames(Step step){
		List<String> names = step.getFeatureNamesIn();

		if(names == null){
			return generateFeatureNames(step);
		}

		return names;
	}

	static
	public List<String> generateFeatureNames(Step step){
		int numberOfFeatures = step.getNumberOfFeatures();

		if(numberOfFeatures == HasNumberOfFeatures.UNKNOWN){
			throw new MissingAttributeException(new Attribute(step, SkLearnFields.N_FEATURES_IN));
		}

		return generateNames("x", numberOfFeatures, true);
	}

	static
	public List<String> generateOutputNames(Estimator estimator){
		int numberOfOutputs = estimator.getNumberOfOutputs();

		if(numberOfOutputs == HasNumberOfOutputs.UNKNOWN){
			throw new MissingAttributeException(new Attribute(estimator, SkLearnFields.N_OUTPUTS));
		}

		return generateNames("y", numberOfOutputs, false);
	}

	static
	private List<String> generateNames(String name, int count, boolean indexed){

		if(count == 1){
			return Collections.singletonList(name + (indexed ? "1" : ""));
		} else

		if(count >= 2){
			List<String> result = new ArrayList<>(count);

			for(int i = 0; i < count; i++){
				result.add(name + String.valueOf(i + 1));
			}

			return result;
		} else

		{
			throw new IllegalArgumentException();
		}
	}

	static
	private Object format(Object object){

		if(object instanceof Attribute){
			Attribute attribute = (Attribute)object;

			ClassDict dict = attribute.getClassDict();
			String name = attribute.getName();

			return dict.getClassName() + "." + name;
		} else

		if(object instanceof ClassDict){
			ClassDict dict = (ClassDict)object;

			return dict.getClassName();
		}

		return object;
	}
}