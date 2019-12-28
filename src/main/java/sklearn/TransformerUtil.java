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
package sklearn;

import java.util.List;

import numpy.DType;
import org.dmg.pmml.DataType;
import org.dmg.pmml.OpType;
import sklearn.pipeline.FeatureUnion;
import sklearn.pipeline.Pipeline;
import sklearn.pipeline.PipelineTransformer;

public class TransformerUtil {

	private TransformerUtil(){
	}

	static
	public int getNumberOfFeatures(Transformer transformer){

		if(transformer instanceof HasNumberOfFeatures){
			HasNumberOfFeatures hasNumberOfFeatures = (HasNumberOfFeatures)transformer;

			return hasNumberOfFeatures.getNumberOfFeatures();
		}

		return HasNumberOfFeatures.UNKNOWN;
	}

	static
	public OpType getOpType(DataType dataType){

		switch(dataType){
			case STRING:
				return OpType.CATEGORICAL;
			case INTEGER:
			case FLOAT:
			case DOUBLE:
				return OpType.CONTINUOUS;
			case BOOLEAN:
				return OpType.CATEGORICAL;
			case DATE:
			case DATE_TIME:
				return OpType.ORDINAL;
			default:
				throw new IllegalArgumentException();
		}
	}

	static
	public OpType getOpType(Object dtype){
		DataType dataType = getDataType(dtype);

		return getOpType(dataType);
	}

	static
	public DataType getDataType(Object dtype){

		if(dtype instanceof String){
			String stringDType = (String)dtype;

			return parseDataType(stringDType);
		} else

		{
			DType numpyDType = (DType)dtype;

			return numpyDType.getDataType();
		}
	}

	static
	public DataType parseDataType(String dtype){

		switch(dtype){
			case "datetime64[D]":
				return DataType.DATE;
			case "datetime64[s]":
				return DataType.DATE_TIME;
			default:
				throw new IllegalArgumentException(dtype);
		}
	}

	static
	public Transformer getHead(List<? extends Transformer> transformers){

		while(transformers.size() > 0){
			Transformer transformer = transformers.get(0);

			if(transformer instanceof FeatureUnion){
				FeatureUnion featureUnion = (FeatureUnion)transformer;

				transformers = featureUnion.getTransformers();
			} else

			if(transformer instanceof PipelineTransformer){
				PipelineTransformer pipelineTransformer = (PipelineTransformer)transformer;

				Pipeline pipeline = pipelineTransformer.getPipeline();

				transformers = pipeline.getTransformers();
			} else

			{
				return transformer;
			}
		}

		return null;
	}
}