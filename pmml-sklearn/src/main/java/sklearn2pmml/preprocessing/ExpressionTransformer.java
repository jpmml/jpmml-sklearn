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
package sklearn2pmml.preprocessing;

import java.util.Collections;
import java.util.List;

import org.dmg.pmml.Apply;
import org.dmg.pmml.DataType;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.Expression;
import org.dmg.pmml.Field;
import org.dmg.pmml.FieldRef;
import org.dmg.pmml.HasDefaultValue;
import org.dmg.pmml.HasMapMissingTo;
import org.dmg.pmml.InvalidValueTreatmentMethod;
import org.dmg.pmml.OpType;
import org.jpmml.converter.ExpressionUtil;
import org.jpmml.converter.Feature;
import org.jpmml.converter.FeatureUtil;
import org.jpmml.converter.TypeUtil;
import org.jpmml.converter.ValueUtil;
import org.jpmml.python.DataFrameScope;
import org.jpmml.python.Scope;
import org.jpmml.python.TypeInfo;
import org.jpmml.sklearn.SkLearnEncoder;
import pandas.CategoricalDtypeUtil;
import pandas.core.CategoricalDtype;
import sklearn.Transformer;
import sklearn2pmml.util.EvaluatableUtil;

public class ExpressionTransformer extends Transformer {

	public ExpressionTransformer(){
		this("sklearn2pmml.preprocessing", "ExpressionTransformer");
	}

	public ExpressionTransformer(String module, String name){
		super(module, name);
	}

	@Override
	public List<Feature> encodeFeatures(List<Feature> features, SkLearnEncoder encoder){
		Object expr = getExpr();
		Object mapMissingTo = getMapMissingTo();
		Object defaultValue = getDefaultValue();
		InvalidValueTreatmentMethod invalidValueTreatment = parseInvalidValueTreatment(getInvalidValueTreatment());
		TypeInfo dtype = getDType();

		if(ValueUtil.isNaN(defaultValue)){
			defaultValue = null;
		} // End if

		if(ValueUtil.isNaN(mapMissingTo)){
			mapMissingTo = null;
		}

		Scope scope = new DataFrameScope("X", features, encoder);

		Expression expression = EvaluatableUtil.translateExpression(expr, scope);

		DerivedField derivedField = null;

		if(expression instanceof FieldRef){
			FieldRef fieldRef = (FieldRef)expression;

			derivedField = encoder.getDerivedField(fieldRef.requireField());

			if(derivedField != null){
				expression = derivedField.getExpression();
			}
		} // End if

		if(mapMissingTo != null){
			HasMapMissingTo<?, Object> hasMapMissingTp = (HasMapMissingTo<?, Object>)expression;

			hasMapMissingTp.setMapMissingTo(mapMissingTo);
		} // End if

		if(defaultValue != null){
			HasDefaultValue<?, Object> hasDefaultValue = (HasDefaultValue<?, Object>)expression;

			hasDefaultValue.setDefaultValue(defaultValue);
		} // End if

		if(invalidValueTreatment != null){
			Apply apply = (Apply)expression;

			apply.setInvalidValueTreatment(invalidValueTreatment);
		}

		DataType dataType;

		if(dtype != null){
			dataType = dtype.getDataType();
		} else

		{
			dataType = ExpressionUtil.getDataType(expression, scope);

			if(dataType == null){
				dataType = DataType.DOUBLE;
			}
		}

		OpType opType = TypeUtil.getOpType(dataType);

		// Detect identity field reference
		if((expression instanceof FieldRef) && (mapMissingTo == null)){
			FieldRef fieldRef = (FieldRef)expression;

			Feature feature = scope.resolveFeature(fieldRef.requireField());
			if(feature != null){
				Field<?> field = feature.getField();

				if((field.requireOpType() == opType) && (field.requireDataType() == dataType)){

					if(dtype instanceof CategoricalDtype){
						CategoricalDtype categoricalDtype = (CategoricalDtype)dtype;

						feature = CategoricalDtypeUtil.refineFeature(feature, categoricalDtype, encoder);
					}

					return Collections.singletonList(feature);
				}
			}
		} // End if

		if(derivedField != null){
			derivedField
				.setOpType(opType)
				.setDataType(dataType);
		} else

		{
			derivedField = encoder.createDerivedField(createFieldName("eval", EvaluatableUtil.toString(expr)), opType, dataType, expression);
		}

		Feature feature = FeatureUtil.createFeature(derivedField, encoder);

		if(dtype instanceof CategoricalDtype){
			CategoricalDtype categoricalDtype = (CategoricalDtype)dtype;

			feature = CategoricalDtypeUtil.refineFeature(feature, categoricalDtype, encoder);
		}

		return Collections.singletonList(feature);
	}

	public Object getDefaultValue(){
		return getOptionalScalar("default_value");
	}

	public ExpressionTransformer setDefaultValue(Object defaultValue){
		put("default_value", defaultValue);

		return this;
	}

	public TypeInfo getDType(){

		// SkLearn2PMML 0.103.2+
		if(containsKey("dtype_")){
			return super.getOptionalDType("dtype_", true);
		}

		// SkLearn2PMML 0.103.1
		return super.getOptionalDType("dtype", true);
	}

	public ExpressionTransformer setDType(Object dtype){
		put("dtype", dtype);

		return this;
	}

	public Object getExpr(){

		// SkLearn2PMML 0.31.0
		if(containsKey("expr_")){
			return getString("expr_");
		} else

		// SkLearn2PMML 0.31.1+
		{
			return getObject("expr");
		}
	}

	public ExpressionTransformer setExpr(String expr){
		put("expr", expr);

		return this;
	}

	public String getInvalidValueTreatment(){
		return getOptionalString("invalid_value_treatment");
	}

	public ExpressionTransformer setInvalidValueTreatment(String invalidValueTreatment){
		put("invalid_value_treatment", invalidValueTreatment);

		return this;
	}

	public Object getMapMissingTo(){
		return getOptionalScalar("map_missing_to");
	}

	public ExpressionTransformer setMapMissingTo(Object mapMissingTo){
		put("map_missing_to", mapMissingTo);

		return this;
	}

	static
	private InvalidValueTreatmentMethod parseInvalidValueTreatment(String invalidValueTreatment){

		if(invalidValueTreatment == null){
			return null;
		}

		switch(invalidValueTreatment){
			case "return_invalid":
				return InvalidValueTreatmentMethod.RETURN_INVALID;
			case "as_missing":
				return InvalidValueTreatmentMethod.AS_MISSING;
			default:
				throw new IllegalArgumentException(invalidValueTreatment);
		}
	}
}