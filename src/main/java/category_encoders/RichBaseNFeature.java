/*
 * Copyright (c) 2021 Villu Ruusmann
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
import java.util.function.Supplier;
import java.util.stream.Collectors;

import com.google.common.collect.Iterables;
import com.google.common.collect.SetMultimap;
import org.dmg.pmml.Apply;
import org.dmg.pmml.DataType;
import org.dmg.pmml.Expression;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.FieldRef;
import org.dmg.pmml.NormDiscrete;
import org.dmg.pmml.PMMLFunctions;
import org.jpmml.converter.BaseNFeature;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.Feature;
import org.jpmml.converter.PMMLEncoder;
import org.jpmml.converter.PMMLUtil;

public class RichBaseNFeature extends BaseNFeature {

	private Object missingCategory = null;


	public RichBaseNFeature(PMMLEncoder encoder, Feature feature, int base, int index, SetMultimap<Integer, ?> values){
		super(encoder, feature.getName(), feature.getDataType(), base, index, values);
	}

	@Override
	public ContinuousFeature toContinuousFeature(){
		FieldName name = getName();
		DataType dataType = getDataType();
		int base = getBase();
		Object missingCategory = getMissingCategory();
		SetMultimap<Integer, ?> values = getValues();

		boolean missingValueAware = values.containsValue(missingCategory);

		Supplier<Expression> expressionSupplier = () -> {
			Map<Integer, ? extends Collection<?>> valueMap = values.asMap();

			if(base == 2){
				Collection<?> categories = valueMap.get(1);

				if(categories != null && categories.size() == 1){
					Object category = Iterables.getOnlyElement(categories);

					if(!missingValueAware){
						return new NormDiscrete(name, category);
					}
				}
			}

			Integer missingBaseValue = 0;

			Apply apply = null;

			Apply prevIfApply = null;

			Collection<? extends Map.Entry<Integer, ? extends Collection<?>>> entries = valueMap.entrySet();

			entries = entries.stream()
				.sorted((left, right) -> Integer.compare(left.getKey(), right.getKey()))
				.filter(entry -> (entry.getKey() > 0))
				.collect(Collectors.toList());

			entries:
			for(Map.Entry<Integer, ? extends Collection<?>> entry : entries){
				Integer baseValue = entry.getKey();
				Collection<?> categories = entry.getValue();

				if(missingValueAware){

					if(categories.contains(missingCategory)){
						categories.remove(missingCategory);

						missingBaseValue = baseValue;
					} // End if

					if(categories.isEmpty()){
						continue entries;
					}
				}

				Apply valueApply = PMMLUtil.createApply((categories.size() == 1 ? PMMLFunctions.EQUAL : PMMLFunctions.ISIN), new FieldRef(name));

				for(Object category : categories){
					valueApply.addExpressions(PMMLUtil.createConstant(category, dataType));
				}

				Apply ifApply = PMMLUtil.createApply(PMMLFunctions.IF, valueApply)
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

				if(missingValueAware){
					apply = PMMLUtil.createApply(PMMLFunctions.IF)
						.addExpressions(PMMLUtil.createApply(PMMLFunctions.ISNOTMISSING, new FieldRef(name)))
						.addExpressions(apply, PMMLUtil.createConstant(missingBaseValue));
				}

				return apply;
			}
		};

		return toContinuousFeature(getDerivedName(), DataType.INTEGER, expressionSupplier);
	}

	public Object getMissingCategory(){
		return this.missingCategory;
	}

	protected void setMissingCategory(Object missingCategory){
		this.missingCategory = missingCategory;
	}
}