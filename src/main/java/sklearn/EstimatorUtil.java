/*
 * Copyright (c) 2015 Villu Ruusmann
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
import java.util.Set;

import com.google.common.base.Function;
import com.google.common.collect.Lists;
import org.dmg.pmml.DataType;
import org.dmg.pmml.DefineFunction;
import org.dmg.pmml.Expression;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.FieldRef;
import org.dmg.pmml.Model;
import org.dmg.pmml.OpType;
import org.dmg.pmml.PMML;
import org.dmg.pmml.ParameterField;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.converter.Schema;
import org.jpmml.sklearn.ClassDictUtil;
import org.jpmml.sklearn.FeatureMapper;

public class EstimatorUtil {

	private EstimatorUtil(){
	}

	static
	public PMML encodePMML(Estimator estimator, FeatureMapper featureMapper){
		Schema schema = estimator.createSchema(featureMapper);

		Set<DefineFunction> defineFunctions = estimator.encodeDefineFunctions();
		for(DefineFunction defineFunction : defineFunctions){
			featureMapper.addDefineFunction(defineFunction);
		}

		Model model = estimator.encodeModel(schema);

		PMML pmml = featureMapper.encodePMML(model);

		return pmml;
	}

	static
	public Classifier asClassifier(Object object){
		return EstimatorUtil.classifierTransformer.apply(object);
	}

	static
	public List<? extends Classifier> asClassifierList(List<?> objects){
		return Lists.transform(objects, EstimatorUtil.classifierTransformer);
	}

	static
	public Regressor asRegressor(Object object){
		return EstimatorUtil.regressorTransformer.apply(object);
	}

	static
	public List<? extends Regressor> asRegressorList(List<?> objects){
		return Lists.transform(objects, EstimatorUtil.regressorTransformer);
	}

	static
	public DefineFunction encodeLogitFunction(){
		return encodeLossFunction("logit", -1d);
	}

	static
	public DefineFunction encodeAdaBoostFunction(){
		return encodeLossFunction("adaboost", -2d);
	}

	static
	private DefineFunction encodeLossFunction(String function, double multiplier){
		FieldName name = FieldName.create("value");

		ParameterField parameterField = new ParameterField(name)
			.setDataType(DataType.DOUBLE)
			.setOpType(OpType.CONTINUOUS);

		// "1 / (1 + exp($multiplier * $name))"
		Expression expression = PMMLUtil.createApply("/", PMMLUtil.createConstant(1d), PMMLUtil.createApply("+", PMMLUtil.createConstant(1d), PMMLUtil.createApply("exp", PMMLUtil.createApply("*", PMMLUtil.createConstant(multiplier), new FieldRef(name)))));

		DefineFunction defineFunction = new DefineFunction(function, OpType.CONTINUOUS, null)
			.setDataType(DataType.DOUBLE)
			.setOpType(OpType.CONTINUOUS)
			.addParameterFields(parameterField)
			.setExpression(expression);

		return defineFunction;
	}

	private static final Function<Object, Classifier> classifierTransformer = new Function<Object, Classifier>(){

		@Override
		public Classifier apply(Object object){

			try {
				if(object == null){
					throw new NullPointerException();
				}

				return (Classifier)object;
			} catch(RuntimeException re){
				throw new IllegalArgumentException("The estimator object (" + ClassDictUtil.formatClass(object) + ") is not a Classifier or is not a supported Classifier subclass", re);
			}
		}
	};

	private static final Function<Object, Regressor> regressorTransformer = new Function<Object, Regressor>(){

		@Override
		public Regressor apply(Object object){

			try {
				if(object == null){
					throw new NullPointerException();
				}

				return (Regressor)object;
			} catch(RuntimeException re){
				throw new IllegalArgumentException("The estimator object (" + ClassDictUtil.formatClass(object) + ") is not a Regressor or is not a supported Regressor subclass", re);
			}
		}
	};
}