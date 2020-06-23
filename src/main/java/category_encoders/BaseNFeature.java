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
package category_encoders;

import java.util.Collection;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.function.Supplier;

import com.google.common.collect.SetMultimap;
import org.dmg.pmml.Apply;
import org.dmg.pmml.DataType;
import org.dmg.pmml.Expression;
import org.dmg.pmml.Field;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.FieldRef;
import org.dmg.pmml.PMMLFunctions;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.Feature;
import org.jpmml.converter.HasDerivedName;
import org.jpmml.converter.PMMLEncoder;
import org.jpmml.converter.PMMLUtil;

abstract
public class BaseNFeature extends Feature implements HasDerivedName {

	private int base = -1;

	private SetMultimap<Integer, ?> values = null;


	public BaseNFeature(PMMLEncoder encoder, Field<?> field, int base, SetMultimap<Integer, ?> values){
		this(encoder, field.getName(), field.getDataType(), base, values);
	}

	public BaseNFeature(PMMLEncoder encoder, Feature feature, int base, SetMultimap<Integer, ?> values){
		this(encoder, feature.getName(), feature.getDataType(), base, values);
	}

	public BaseNFeature(PMMLEncoder encoder, FieldName name, DataType dataType, int base, SetMultimap<Integer, ?> values){
		super(encoder, name, dataType);

		setBase(base);
		setValues(values);
	}

	@Override
	public ContinuousFeature toContinuousFeature(){
		FieldName name = getName();
		DataType dataType = getDataType();
		SetMultimap<Integer, ?> values = getValues();

		Supplier<Expression> expressionSupplier = () -> {
			Apply apply = null;

			Apply prevIfApply = null;

			Set<? extends Map.Entry<Integer, ? extends Collection<?>>> entries = (values.asMap()).entrySet();
			for(Map.Entry<Integer, ? extends Collection<?>> entry : entries){
				Integer baseValue = entry.getKey();
				Collection<?> categories = entry.getValue();

				Apply isInApply = PMMLUtil.createApply(PMMLFunctions.ISIN, new FieldRef(name));

				for(Object category : categories){
					isInApply.addExpressions(PMMLUtil.createConstant(category, dataType));
				}

				Apply ifApply = PMMLUtil.createApply(PMMLFunctions.IF, isInApply)
					.addExpressions(PMMLUtil.createConstant(baseValue));

				if(apply == null){
					apply = ifApply;
				} // End if

				if(prevIfApply != null){
					prevIfApply.addExpressions(ifApply);
				}

				prevIfApply = ifApply;
			}

			if(apply == null){
				return PMMLUtil.createConstant(0);
			} else

			{
				prevIfApply.addExpressions(PMMLUtil.createConstant(0));

				return apply;
			}
		};

		return toContinuousFeature(getDerivedName(), DataType.INTEGER, expressionSupplier);
	}

	@Override
	public int hashCode(){
		int result = super.hashCode();

		result = (31 * result) + Objects.hash(this.getBase());
		result = (31 * result) + Objects.hash(this.getValues());

		return result;
	}

	@Override
	public boolean equals(Object object){

		if(object instanceof BaseNFeature){
			BaseNFeature that = (BaseNFeature)object;

			return super.equals(object) && Objects.equals(this.getBase(), that.getBase()) && Objects.equals(this.getValues(), that.getValues());
		}

		return false;
	}

	public int getBase(){
		return this.base;
	}

	private void setBase(int base){
		this.base = base;
	}

	public SetMultimap<Integer, ?> getValues(){
		return this.values;
	}

	private void setValues(SetMultimap<Integer, ?> values){
		this.values = Objects.requireNonNull(values);
	}
}