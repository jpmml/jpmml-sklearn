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

import org.dmg.pmml.Apply;
import org.dmg.pmml.DataType;
import org.dmg.pmml.DefineFunction;
import org.dmg.pmml.Expression;
import org.dmg.pmml.FieldRef;
import org.dmg.pmml.OpType;
import org.dmg.pmml.PMMLFunctions;
import org.dmg.pmml.ParameterField;
import org.jpmml.converter.ExpressionUtil;
import org.jpmml.converter.Feature;
import org.jpmml.python.TypeInfo;
import org.jpmml.sklearn.SkLearnEncoder;

public class TfidfVectorizer extends CountVectorizer {

	public TfidfVectorizer(String module, String name){
		super(module, name);
	}

	@Override
	public TypeInfo getDType(){
		TypeInfo dtype = super.getDType();

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
	public DefineFunction encodeDefineFunction(Feature feature, SkLearnEncoder encoder){
		TfidfTransformer transformer = getTransformer();

		Boolean sublinearTf = transformer.getSublinearTf();
		Boolean useIdf = transformer.getUseIdf();

		DefineFunction defineFunction = super.encodeDefineFunction(feature, encoder);

		if(!(sublinearTf || useIdf)){
			return defineFunction;
		}

		Expression expression = defineFunction.requireExpression();

		if(sublinearTf){
			DefineFunction sublinearDefineFunction = ensureSublinearDefineFunction(encoder);

			expression = ExpressionUtil.createApply(sublinearDefineFunction, expression);
		} // End if

		if(useIdf){
			ParameterField weightField = new ParameterField("weight");

			defineFunction.addParameterFields(weightField);

			expression = ExpressionUtil.createApply(PMMLFunctions.MULTIPLY, expression, new FieldRef(weightField));
		}

		defineFunction
			.setDataType(DataType.DOUBLE)
			.setExpression(expression);

		return defineFunction;
	}

	@Override
	public Apply encodeApply(DefineFunction defineFunction, Feature feature, int index, String term){
		TfidfTransformer transformer = getTransformer();

		Apply apply = super.encodeApply(defineFunction, feature, index, term);

		Boolean useIdf = transformer.getUseIdf();
		if(useIdf){
			Number weight = transformer.getWeight(index);

			apply.addExpressions(ExpressionUtil.createConstant(weight));
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
		return get("_tfidf", TfidfTransformer.class);
	}

	static
	private DefineFunction ensureSublinearDefineFunction(SkLearnEncoder encoder){
		DefineFunction defineFunction = encoder.getDefineFunction("sublinearize");

		if(defineFunction == null){
			ParameterField valueField = new ParameterField("x");

			Apply apply = ExpressionUtil.createApply(PMMLFunctions.IF,
				ExpressionUtil.createApply(PMMLFunctions.GREATERTHAN, new FieldRef(valueField), ExpressionUtil.createConstant(0)),
				ExpressionUtil.createApply(PMMLFunctions.ADD, ExpressionUtil.createApply(PMMLFunctions.LN, new FieldRef(valueField)), ExpressionUtil.createConstant(1)),
				ExpressionUtil.createConstant(0)
			);

			defineFunction = new DefineFunction("sublinearize", OpType.CONTINUOUS, DataType.DOUBLE, null, apply)
				.addParameterFields(valueField);

			encoder.addDefineFunction(defineFunction);
		}

		return defineFunction;
	}
}