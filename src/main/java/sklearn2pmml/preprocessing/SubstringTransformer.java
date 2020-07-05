/*
 * Copyright (c) 2019 Villu Ruusmann
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

import java.util.Collections;
import java.util.List;

import org.dmg.pmml.Apply;
import org.dmg.pmml.DataType;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.OpType;
import org.dmg.pmml.PMMLFunctions;
import org.jpmml.converter.Feature;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.converter.StringFeature;
import org.jpmml.python.ClassDictUtil;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn.Transformer;

public class SubstringTransformer extends Transformer {

	public SubstringTransformer(String module, String name){
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
		Integer begin = getBegin();
		Integer end = getEnd();

		if((begin < 0) || (end < begin)){
			throw new IllegalArgumentException();
		}

		ClassDictUtil.checkSize(1, features);

		Feature feature = features.get(0);

		if(!(DataType.STRING).equals(feature.getDataType())){
			throw new IllegalArgumentException();
		}

		Apply apply = PMMLUtil.createApply(PMMLFunctions.SUBSTRING)
			.addExpressions(feature.ref())
			.addExpressions(PMMLUtil.createConstant(begin + 1, DataType.INTEGER), PMMLUtil.createConstant((end - begin), DataType.INTEGER));

		DerivedField derivedField = encoder.createDerivedField(createFieldName("substring", feature), OpType.CATEGORICAL, DataType.STRING, apply);

		return Collections.singletonList(new StringFeature(encoder, derivedField));
	}

	public Integer getBegin(){
		return getInteger("begin");
	}

	public Integer getEnd(){
		return getInteger("end");
	}
}