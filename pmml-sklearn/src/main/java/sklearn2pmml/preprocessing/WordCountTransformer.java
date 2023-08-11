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
package sklearn2pmml.preprocessing;

import java.util.List;

import org.dmg.pmml.DataType;
import org.dmg.pmml.OpType;
import org.jpmml.converter.Feature;
import org.jpmml.python.ClassDictUtil;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn.Transformer;
import sklearn.pipeline.SkLearnPipeline;

public class WordCountTransformer extends Transformer {

	public WordCountTransformer(String module, String name){
		super(module, name);
	}

	@Override
	public OpType getOpType(){
		return OpType.CATEGORICAL;
	}

	@Override
	public DataType getDataType(){
		return DataType.STRING;
	}

	@Override
	public List<Feature> encodeFeatures(List<Feature> features, SkLearnEncoder encoder){
		SkLearnPipeline pipeline = getPipeline();

		ClassDictUtil.checkSize(1, features);

		Transformer transformer = pipeline.toTransformer();

		return transformer.encode(features, encoder);
	}

	public SkLearnPipeline getPipeline(){
		return get("pipeline_", SkLearnPipeline.class);
	}
}