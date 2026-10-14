/*
 * Copyright (c) 2020 Villu Ruusmann
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
package sklearn;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import org.dmg.pmml.DataType;
import org.dmg.pmml.OpType;
import org.dmg.pmml.PMML;
import org.jpmml.converter.ExceptionUtil;
import org.jpmml.converter.Feature;
import org.jpmml.converter.PMMLEncodable;
import org.jpmml.converter.SchemaException;
import org.jpmml.python.Attribute;
import org.jpmml.python.MissingAttributeException;
import sklearn.pipeline.SkLearnPipeline;

public class StepUtil {

	private StepUtil(){
	}

	static
	public Step getHead(Step step){

		if(step instanceof HasHead){
			HasHead hasHead = (HasHead)step;

			return hasHead.getHead();
		}

		return step;
	}

	static
	public Step getTail(Step step){

		if(step instanceof HasTail){
			HasTail hasTail = (HasTail)step;

			return hasTail.getTail();
		}

		return step;
	}

	static
	public int getNumberOfFeatures(List<? extends Step> steps){

		for(Step step : steps){
			return step.getNumberOfFeatures();
		}

		return HasNumberOfFeatures.UNKNOWN;
	}

	static
	public void checkNumberOfFeatures(Step step, List<? extends Feature> features){
		int numberOfFeatures = step.getNumberOfFeatures();

		if((numberOfFeatures != HasNumberOfFeatures.UNKNOWN) && (numberOfFeatures != features.size())){
			throw new SchemaException("Expected " + ExceptionUtil.formatCount(numberOfFeatures, "feature") + ", got " + features.size());
		}
	}

	static
	public Object getTag(Map<String, ?> tags, String name){
		return getTag(tags, name, null);
	}

	static
	public Object getTag(Map<String, ?> tags, String name, Object defaultValue){

		if(tags != null){
			return ((Map)tags).getOrDefault(name, defaultValue);
		}

		return defaultValue;
	}

	static
	public HasMultiType getType(Step step){

		if(step instanceof HasMultiType){
			HasMultiType hasMultiType = (HasMultiType)step;

			return hasMultiType;
		} else

		{
			HasMultiType hasMultiType = new HasMultiType(){

				private OpType opType = null;

				private DataType dataType = null;


				@Override
				public OpType getOpType(){

					if(this.opType == null){
						this.opType = step.getOpType();
					}

					return this.opType;
				}

				@Override
				public DataType getDataType(){

					if(this.dataType == null){
						this.dataType = step.getDataType();
					}

					return this.dataType;
				}
			};

			return hasMultiType;
		}
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

	/**
	 * @see EstimatorUtil#encodePMML(Estimator)
	 * @see TransformerUtil#encodePMML(Transformer)
	 */
	static
	public <E extends Step & PMMLEncodable> PMML encodePMML(E step, Map<String, ?> options){

		if(!options.isEmpty()){
			Estimator estimator = getFinalEstimator(step);

			if(estimator != null){
				Map<String, ?> prevPmmlOptions = estimator.getPMMLOptions();

				Map<String, Object> pmmlOptions = new LinkedHashMap<>();

				if(prevPmmlOptions != null){
					pmmlOptions.putAll(prevPmmlOptions);
				}

				pmmlOptions.putAll(options);

				try {
					estimator.setPMMLOptions(pmmlOptions);

					return step.encodePMML(options);
				} finally {
					estimator.setPMMLOptions(prevPmmlOptions);
				}
			}
		}

		return step.encodePMML(options);
	}

	static
	private Estimator getFinalEstimator(Step step){

		if(step instanceof SkLearnPipeline){
			SkLearnPipeline pipeline = (SkLearnPipeline)step;

			if(pipeline.hasFinalEstimator()){
				Step tail = pipeline.getTail();

				if(tail instanceof Estimator){
					Estimator estimator = (Estimator)tail;

					return estimator;
				}
			}
		} else

		if(step instanceof Estimator){
			Estimator estimator = (Estimator)step;

			return estimator;
		}

		return null;
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
}