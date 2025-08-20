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

import java.util.List;
import java.util.Map;

import org.dmg.pmml.DataType;
import org.dmg.pmml.OpType;
import org.jpmml.converter.Feature;
import org.jpmml.python.ClassDictUtil;

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
			throw new IllegalArgumentException("Expected " + numberOfFeatures + " feature(s) (" + ClassDictUtil.formatClass(step)  + "), got " + features.size() + " feature(s)");
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
}