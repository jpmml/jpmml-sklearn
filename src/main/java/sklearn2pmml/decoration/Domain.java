/*
 * Copyright (c) 2016 Villu Ruusmann
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
package sklearn2pmml.decoration;

import java.time.LocalDate;
import java.time.LocalDateTime;
import java.util.Collection;
import java.util.Collections;
import java.util.GregorianCalendar;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import com.google.common.base.Function;
import com.google.common.collect.Lists;
import net.razorvine.pickle.objects.ClassDict;
import org.dmg.pmml.Counts;
import org.dmg.pmml.DataField;
import org.dmg.pmml.DataType;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.InvalidValueTreatmentMethod;
import org.dmg.pmml.MissingValueTreatmentMethod;
import org.dmg.pmml.OpType;
import org.dmg.pmml.Value;
import org.jpmml.converter.Feature;
import org.jpmml.converter.InvalidValueDecorator;
import org.jpmml.converter.MissingValueDecorator;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.converter.WildcardFeature;
import org.jpmml.python.CalendarUtil;
import org.jpmml.python.ClassDictUtil;
import org.jpmml.python.HasArray;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn.Transformer;

abstract
public class Domain extends Transformer {

	public Domain(String module, String name){
		super(module, name);
	}

	@Override
	public List<Feature> encodeFeatures(List<Feature> features, SkLearnEncoder encoder){
		MissingValueTreatmentMethod missingValueTreatment = DomainUtil.parseMissingValueTreatment(getMissingValueTreatment());
		Object missingValueReplacement = getMissingValueReplacement();
		List<?> missingValues = getMissingValues();

		if(missingValueReplacement != null){

			if(missingValueTreatment == null){
				missingValueTreatment = MissingValueTreatmentMethod.AS_VALUE;
			}
		}

		InvalidValueTreatmentMethod invalidValueTreatment = DomainUtil.parseInvalidValueTreatment(getInvalidValueTreatment());
		Object invalidValueReplacement = getInvalidValueReplacement();

		if(invalidValueReplacement != null){

			if(invalidValueTreatment == null){
				invalidValueTreatment = InvalidValueTreatmentMethod.AS_IS;
			}
		}

		for(Feature feature : features){
			WildcardFeature wildcardFeature = asWildcardFeature(feature);

			DataField dataField = wildcardFeature.getField();

			DataType dataType = dataField.getDataType();

			if(missingValueTreatment != null){
				Object pmmlMissingValueReplacement = (missingValueReplacement != null ? standardizeValue(dataType, missingValueReplacement) : null);

				encoder.addDecorator(dataField, new MissingValueDecorator(missingValueTreatment, pmmlMissingValueReplacement));
			} // End if

			if(missingValues != null){
				PMMLUtil.addValues(dataField, standardizeValues(dataType, missingValues), Value.Property.MISSING);
			} // End if

			if(invalidValueTreatment != null){
				Object pmmlInvalidValueReplacement = (invalidValueReplacement != null ? standardizeValue(dataType, invalidValueReplacement) : null);

				encoder.addDecorator(dataField, new InvalidValueDecorator(invalidValueTreatment, pmmlInvalidValueReplacement));
			}
		}

		return features;
	}

	@Override
	public DataField updateDataField(DataField dataField, OpType opType, DataType dataType, SkLearnEncoder encoder){
		FieldName name = dataField.getName();

		if(encoder.isFrozen(name)){
			throw new IllegalArgumentException("Field " + name.getValue() + " is frozen for type information updates");
		}

		dataField
			.setDataType(dataType)
			.setOpType(opType);

		encoder.setDomain(name, this);

		return dataField;
	}

	public Object getDType(){
		Object dtype = get("dtype");

		if(dtype == null){
			return null;
		}

		return super.getDType(true);
	}

	public String getMissingValueTreatment(){
		return getOptionalString("missing_value_treatment");
	}

	public Object getMissingValueReplacement(){
		return getOptionalScalar("missing_value_replacement");
	}

	public List<Object> getMissingValues(){
		Object missingValues = getOptionalObject("missing_values");

		if(missingValues != null){

			// SkLearn2PMML 0.38.0
			if(!(missingValues instanceof List)){
				return Collections.singletonList(missingValues);
			}

			// SkLearn2PMML 0.38.1+
			return (List)missingValues;
		}

		return null;
	}

	public String getInvalidValueTreatment(){
		return getOptionalString("invalid_value_treatment");
	}

	public Object getInvalidValueReplacement(){
		return getOptionalScalar("invalid_value_replacement");
	}

	public Boolean getWithData(){
		return getOptionalBoolean("with_data", Boolean.TRUE);
	}

	public Boolean getWithStatistics(){
		return getOptionalBoolean("with_statistics", Boolean.FALSE);
	}

	public Map<String, ?> getCounts(){
		return getDict("counts_");
	}

	static
	public Counts createCounts(Map<String, ?> values){
		Counts counts = new Counts()
			.setTotalFreq(selectValue(values, "totalFreq", 0))
			.setMissingFreq(selectValue(values, "missingFreq"))
			.setInvalidFreq(selectValue(values, "invalidFreq"));

		return counts;
	}

	static
	protected Map<String, ?> extractMap(Map<String, ?> map, int index){
		Map<String, Object> result = new LinkedHashMap<>();

		Collection<? extends Map.Entry<String, ?>> entries = map.entrySet();
		for(Map.Entry<String, ?> entry : entries){
			String key = entry.getKey();
			List<?> value = asArray(entry.getValue());

			result.put(key, value.get(index));
		}

		return result;
	}

	static
	protected Number selectValue(Map<String, ?> values, String key){
		return selectValue(values, key, null);
	}

	static
	protected Number selectValue(Map<String, ?> values, String key, Number defaultValue){
		Number value = (Number)values.get(key);

		if(value == null){
			return defaultValue;
		}

		return value;
	}

	static
	protected WildcardFeature asWildcardFeature(Feature feature){

		if(feature instanceof WildcardFeature){
			WildcardFeature wildcardFeature = (WildcardFeature)feature;

			return wildcardFeature;
		}

		throw new IllegalArgumentException("Field " + feature.getName() + " is not decorable");
	}

	static
	protected List<?> asArray(Object object){

		if(object instanceof HasArray){
			HasArray hasArray = (HasArray)object;

			return hasArray.getArrayContent();
		} // End if

		if(object instanceof Number){
			return Collections.singletonList(object);
		}

		throw new IllegalArgumentException(ClassDictUtil.formatClass(object) + " is not a supported array type");
	}

	static
	protected Object standardizeValue(DataType dataType, Object value){

		switch(dataType){
			case DATE:
				return CalendarUtil.toLocalDate((GregorianCalendar)value);
			case DATE_TIME:
				return CalendarUtil.toLocalDateTime((GregorianCalendar)value);
			default:
				return checkValue(value);
		}
	}

	static
	protected List<?> standardizeValues(DataType dataType, List<?> values){
		Function<Object, Object> function;

		switch(dataType){
			case DATE:
				function = new Function<Object, Object>(){

					@Override
					public LocalDate apply(Object object){
						return CalendarUtil.toLocalDate((GregorianCalendar)object);
					}
				};
				break;
			case DATE_TIME:
				function = new Function<Object, Object>(){

					@Override
					public LocalDateTime apply(Object object){
						return CalendarUtil.toLocalDateTime((GregorianCalendar)object);
					}
				};
				break;
			default:
				function = new Function<Object, Object>(){

					@Override
					public Object apply(Object object){
						return checkValue(object);
					}
				};
				break;
		}

		return Lists.transform(values, function);
	}

	static
	public Object checkValue(Object object){

		if(object instanceof ClassDict){
			ClassDict classDict = (ClassDict)object;

			throw new IllegalArgumentException("Expected Java primitive value, got Python class " + classDict.getClassName() + " value");
		}

		return object;
	}
}