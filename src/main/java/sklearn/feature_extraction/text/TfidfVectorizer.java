/*
 * Copyright (c) 2017 Villu Ruusmann
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
package sklearn.feature_extraction.text;

import java.util.List;

import numpy.DType;
import org.dmg.pmml.Apply;
import org.dmg.pmml.DataType;
import org.dmg.pmml.DefineFunction;
import org.dmg.pmml.Expression;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.FieldRef;
import org.dmg.pmml.ParameterField;
import org.jpmml.converter.Feature;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.sklearn.SkLearnEncoder;

public class TfidfVectorizer extends CountVectorizer {

	public TfidfVectorizer(String module, String name){
		super(module, name);
	}

	@Override
	public DType getDType(){
		DType dtype = super.getDType();

		if(dtype != null){
			TfidfTransformer transformer = getTransformer();

			if(transformer != null){
				DataType dataType = dtype.getDataType();

				switch(dataType){
					case BOOLEAN:
					case INTEGER:
						return null;
					case FLOAT:
					case DOUBLE:
						return dtype;
					default:
						break;
				}
			}
		}

		return dtype;
	}

	@Override
	public List<Feature> encodeFeatures(List<Feature> features, SkLearnEncoder encoder){
		TfidfTransformer transformer = getTransformer();

		String norm = transformer.getNorm();
		if(norm != null){
			throw new IllegalArgumentException(norm);
		}

		return super.encodeFeatures(features, encoder);
	}

	@Override
	public DefineFunction encodeDefineFunction(){
		TfidfTransformer transformer = getTransformer();

		DefineFunction defineFunction = super.encodeDefineFunction();

		Expression expression = defineFunction.getExpression();

		Boolean sublinearTf = transformer.getSublinearTf();
		if(sublinearTf){
			expression = PMMLUtil.createApply("+", PMMLUtil.createApply("log", expression), PMMLUtil.createConstant(1d));
		} // End if

		Boolean useIdf = transformer.getUseIdf();
		if(useIdf){
			ParameterField weight = new ParameterField(FieldName.create("weight"));

			defineFunction.addParameterFields(weight);

			expression = PMMLUtil.createApply("*", expression, new FieldRef(weight.getName()));
		}

		defineFunction.setExpression(expression);

		return defineFunction;
	}

	@Override
	public Apply encodeApply(String function, Feature feature, int index, String term){
		TfidfTransformer transformer = getTransformer();

		Apply apply = super.encodeApply(function, feature, index, term);

		Boolean useIdf = transformer.getUseIdf();
		if(useIdf){
			Number weight = transformer.getWeight(index);

			apply.addExpressions(PMMLUtil.createConstant(weight));
		}

		return apply;
	}

	@Override
	public String functionName(){
		TfidfTransformer transformer = getTransformer();

		Boolean useIdf = transformer.getUseIdf();
		if(useIdf){
			return "tf-idf";
		}

		return super.functionName();
	}

	public TfidfTransformer getTransformer(){
		return (TfidfTransformer)get("_tfidf");
	}
}