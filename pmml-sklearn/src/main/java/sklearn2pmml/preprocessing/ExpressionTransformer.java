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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

import org.dmg.pmml.Apply;
import org.dmg.pmml.Constant;
import org.dmg.pmml.DataType;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.Expression;
import org.dmg.pmml.Field;
import org.dmg.pmml.FieldRef;
import org.dmg.pmml.HasDefaultValue;
import org.dmg.pmml.HasInvalidValueTreatment;
import org.dmg.pmml.HasMapMissingTo;
import org.dmg.pmml.InvalidValueTreatmentMethod;
import org.dmg.pmml.OpType;
import org.jpmml.converter.ExpressionUtil;
import org.jpmml.converter.Feature;
import org.jpmml.converter.FeatureUtil;
import org.jpmml.converter.TypeUtil;
import org.jpmml.converter.ValueUtil;
import org.jpmml.python.ClassDictUtil;
import org.jpmml.python.DataFrameScope;
import org.jpmml.python.Scope;
import org.jpmml.python.TypeInfo;
import org.jpmml.sklearn.SkLearnEncoder;
import org.jpmml.sklearn.SkLearnException;
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

		udf:
		if(expression instanceof Apply){
			Apply apply = (Apply)expression;

			if(isConstantLike(apply)){
				break udf;
			}

			List<String> attributes = new ArrayList<>();

			if(mapMissingTo != null){
				attributes.add(ClassDictUtil.formatMember(this, "map_missing_to"));
			} // End if

			if(defaultValue != null){
				attributes.add(ClassDictUtil.formatMember(this, "default_value"));
			} // End if

			if(invalidValueTreatment != null){
				attributes.add(ClassDictUtil.formatMember(this, "invalid_value_treatment"));
			} // End if

			if(!attributes.isEmpty()){
				List<String> quotedAttributes = attributes.stream()
					.map(name -> "\'" + name + "\'")
					.collect(Collectors.toList());

				throw new SkLearnException("The target PMML element for " + String.join(", ", quotedAttributes) + " attribute(s) is unclear. Please refactor the expression from inline string representation to UDF representation");
			}
		} // End if

		if(mapMissingTo != null){
			HasMapMissingTo<?, Object> hasMapMissingTo = (HasMapMissingTo<?, Object>)expression;

			hasMapMissingTo.setMapMissingTo(mapMissingTo);
		} // End if

		if(defaultValue != null){
			HasDefaultValue<?, Object> hasDefaultValue = (HasDefaultValue<?, Object>)expression;

			hasDefaultValue.setDefaultValue(defaultValue);
		} // End if

		if(invalidValueTreatment != null){
			HasInvalidValueTreatment hasInvalidValueTreatment = (HasInvalidValueTreatment)expression;

			hasInvalidValueTreatment.setInvalidValueTreatment(invalidValueTreatment);
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
		setattr("default_value", defaultValue);

		return this;
	}

	public TypeInfo getDType(){

		// SkLearn2PMML 0.103.2+
		if(hasattr("dtype_")){
			return super.getOptionalDType("dtype_", true);
		}

		// SkLearn2PMML 0.103.1
		return super.getOptionalDType("dtype", true);
	}

	public ExpressionTransformer setDType(Object dtype){
		setattr("dtype", dtype);

		return this;
	}

	public Object getExpr(){

		// SkLearn2PMML 0.31.0
		if(hasattr("expr_")){
			return getString("expr_");
		} else

		// SkLearn2PMML 0.31.1+
		{
			return getObject("expr");
		}
	}

	public ExpressionTransformer setExpr(String expr){
		setattr("expr", expr);

		return this;
	}

	public String getInvalidValueTreatment(){
		return getOptionalEnum("invalid_value_treatment", this::getOptionalString, Arrays.asList(ExpressionTransformer.INVALIDVALUETREATMENT_AS_MISSING, ExpressionTransformer.INVALIDVALUETREATMENT_RETURN_INVALID));
	}

	public ExpressionTransformer setInvalidValueTreatment(String invalidValueTreatment){
		setattr("invalid_value_treatment", invalidValueTreatment);

		return this;
	}

	public Object getMapMissingTo(){
		return getOptionalScalar("map_missing_to");
	}

	public ExpressionTransformer setMapMissingTo(Object mapMissingTo){
		setattr("map_missing_to", mapMissingTo);

		return this;
	}

	static
	private InvalidValueTreatmentMethod parseInvalidValueTreatment(String invalidValueTreatment){

		if(invalidValueTreatment == null){
			return null;
		}

		switch(invalidValueTreatment){
			case ExpressionTransformer.INVALIDVALUETREATMENT_AS_MISSING:
				return InvalidValueTreatmentMethod.AS_MISSING;
			case ExpressionTransformer.INVALIDVALUETREATMENT_RETURN_INVALID:
				return InvalidValueTreatmentMethod.RETURN_INVALID;
			default:
				throw new IllegalArgumentException(invalidValueTreatment);
		}
	}

	static
	private boolean isConstantLike(Apply apply){

		if(apply.hasExpressions()){
			List<Expression> expressions = apply.getExpressions();

			for(Expression expression : expressions){

				if(expression instanceof Constant){
					Constant constant = (Constant)expression;
				} else

				if(expression instanceof FieldRef){
					FieldRef fieldRef = (FieldRef)expression;
				} else

				{
					return false;
				}
			}
		}

		return true;
	}

	private static final String INVALIDVALUETREATMENT_AS_MISSING = "as_missing";
	private static final String INVALIDVALUETREATMENT_RETURN_INVALID = "return_invalid";
}