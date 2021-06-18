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
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.function.Predicate;
import java.util.function.Supplier;

import org.dmg.pmml.DataType;
import org.dmg.pmml.Expression;
import org.dmg.pmml.Field;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.MapValues;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.Feature;
import org.jpmml.converter.FieldNameUtil;
import org.jpmml.converter.PMMLEncoder;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.converter.ThresholdFeature;
import org.jpmml.converter.TypeUtil;
import org.jpmml.model.ToStringHelper;

public class MapFeature extends ThresholdFeature {

	private Map<?, ? extends Number> mapping = null;

	private Object missingValue = null;


	public MapFeature(PMMLEncoder encoder, Feature feature, Map<?, ? extends Number> mapping, Object missingValue){
		this(encoder, feature.getName(), feature.getDataType(), mapping, missingValue);
	}

	public MapFeature(PMMLEncoder encoder, Field<?> field, Map<?, ? extends Number> mapping, Object missingValue){
		this(encoder, field.getName(), field.getDataType(), mapping, missingValue);
	}

	public MapFeature(PMMLEncoder encoder, FieldName name, DataType dataType, Map<?, ? extends Number> mapping, Object missingValue){
		super(encoder, name, dataType);

		setMapping(mapping);
		setMissingValue(missingValue);
	}

	@Override
	public FieldName getDerivedName(){
		return FieldNameUtil.create("map", getName());
	}

	@Override
	public ContinuousFeature toContinuousFeature(){
		FieldName name = getName();
		Map<?, ? extends Number> mapping = getMapping();
		Object missingValue = getMissingValue();

		Supplier<Expression> expressionSupplier = () -> {
			Map<?, ? extends Number> validMapping = new LinkedHashMap<>(mapping);

			Number mapMissingTo = validMapping.remove(missingValue);

			MapValues mapValues = PMMLUtil.createMapValues(name, validMapping)
				.setMapMissingTo(mapMissingTo);

			return mapValues;
		};

		DataType dataType = TypeUtil.getDataType(mapping.values(), DataType.DOUBLE);

		return toContinuousFeature(getDerivedName(), dataType, expressionSupplier);
	}

	@Override
	public Set<?> getValues(Predicate<Number> predicate){
		Map<?, ? extends Number> mapping = getMapping();

		Set<Object> result = new LinkedHashSet<>();

		Collection<? extends Map.Entry<?, ? extends Number>> entries = mapping.entrySet();

		entries.stream()
			.filter(entry -> predicate.test(entry.getValue()))
			.map(entry -> entry.getKey())
			.forEach(result::add);

		return result;
	}

	@Override
	public int hashCode(){
		int result = super.hashCode();

		result = (31 * result) + Objects.hash(this.getMapping());
		result = (31 * result) + Objects.hash(this.getMissingValue());

		return result;
	}

	@Override
	public boolean equals(Object object){

		if(object instanceof MapFeature){
			MapFeature that = (MapFeature)object;

			return super.equals(object) && Objects.equals(this.getMapping(), that.getMapping()) && Objects.equals(this.getMissingValue(), that.getMissingValue());
		}

		return false;
	}

	@Override
	protected ToStringHelper toStringHelper(){
		return new ToStringHelper(this)
			.add("mapping", getMapping())
			.add("missingValue", getMissingValue());
	}

	public Map<?, ? extends Number> getMapping(){
		return this.mapping;
	}

	private void setMapping(Map<?, ? extends Number> mapping){
		this.mapping = Objects.requireNonNull(mapping);
	}

	@Override
	public Object getMissingValue(){
		return this.missingValue;
	}

	private void setMissingValue(Object missingValue){
		this.missingValue = missingValue;
	}
}