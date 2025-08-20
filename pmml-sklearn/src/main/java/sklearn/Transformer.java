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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

import builtins.TypeConstructor;
import com.google.common.collect.Lists;
import org.dmg.pmml.DataField;
import org.dmg.pmml.DataType;
import org.dmg.pmml.OpType;
import org.jpmml.converter.Feature;
import org.jpmml.converter.FieldNameUtil;
import org.jpmml.converter.WildcardFeature;
import org.jpmml.python.AttributeCastFunction;
import org.jpmml.python.CastFunction;
import org.jpmml.python.ClassDictUtil;
import org.jpmml.python.PythonTypeUtil;
import org.jpmml.python.TypeInfo;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn2pmml.HasPMMLName;

abstract
public class Transformer extends Step implements HasPMMLName<Transformer> {

	public Transformer(String module, String name){
		super(module, name);
	}

	abstract
	public List<Feature> encodeFeatures(List<Feature> features, SkLearnEncoder encoder);

	@Override
	public int getNumberOfFeatures(){

		if(hasattr(SkLearnFields.N_FEATURES_IN)){
			return getInteger(SkLearnFields.N_FEATURES_IN);
		}

		return HasNumberOfFeatures.UNKNOWN;
	}

	@Override
	public OpType getOpType(){
		return OpType.CONTINUOUS;
	}

	@Override
	public DataType getDataType(){
		return DataType.DOUBLE;
	}

	public Map<String, ?> getTransformerTags(){
		return (Map)StepUtil.getTag(getSkLearnTags(), "transformer_tags");
	}

	public List<Feature> encode(List<Feature> features, SkLearnEncoder encoder){
		checkVersion();

		checkFeatures(features);

		features = updateFeatures(features, encoder);

		return encodeFeatures(features, encoder);
	}

	public void checkFeatures(List<? extends Feature> features){
		StepUtil.checkNumberOfFeatures(this, features);
	}

	public List<Feature> updateFeatures(List<Feature> features, SkLearnEncoder encoder){
		HasMultiType hasMultiType = StepUtil.getType(this);

		List<Feature> result = new ArrayList<>(features.size());

		for(int i = 0; i < features.size(); i++){
			Feature feature = features.get(i);

			update:
			if(feature instanceof WildcardFeature){
				WildcardFeature wildcardFeature = (WildcardFeature)feature;

				OpType opType;
				DataType dataType;

				try {
					opType = hasMultiType.getOpType(i);
					dataType = hasMultiType.getDataType(i);
				} catch(UnsupportedOperationException uoe){
					break update;
				}

				feature = refineWildcardFeature(wildcardFeature, opType, dataType, encoder);
			}

			result.add(feature);
		}

		return result;
	}

	public WildcardFeature refineWildcardFeature(WildcardFeature wildcardFeature, OpType opType, DataType dataType, SkLearnEncoder encoder){
		String name = wildcardFeature.getName();

		DataField dataField = encoder.getDataField(name);
		if(dataField == null){
			throw new IllegalArgumentException("Field " + name + " is undefined");
		} // End if

		if((dataField.requireOpType() != opType) || (dataField.requireDataType() != dataType)){
			dataField = updateDataField(dataField, opType, dataType, encoder);

			wildcardFeature = new WildcardFeature(encoder, dataField);
		}

		return wildcardFeature;
	}

	public DataField updateDataField(DataField dataField, OpType opType, DataType dataType, SkLearnEncoder encoder){
		String name = dataField.requireName();

		if(encoder.isFrozen(name)){
			return dataField;
		}

		switch(dataType){
			case DOUBLE:
				// If the DataField element already specifies a non-default data type, then keep it
				if(dataField.requireDataType() != DataType.DOUBLE){
					dataType = dataField.requireDataType();
				}
				break;
		}

		dataField
			.setOpType(opType)
			.setDataType(dataType);

		return dataField;
	}

	public TypeInfo getDType(String name, boolean extended){
		Object dtype = getObject(name);

		CastFunction<TypeInfo> castFunction = new AttributeCastFunction<TypeInfo>(TypeInfo.class){

			@Override
			public TypeInfo apply(Object object){
				object = toTypeInfo(object, extended);

				return super.apply(object);
			}

			@Override
			protected String formatMessage(Object object){
				return "Data type attribute \'" + ClassDictUtil.formatMember(Transformer.this, name) + "\' has an unsupported value (" + ClassDictUtil.formatClass(object) + ")";
			}
		};

		return castFunction.apply(dtype);
	}

	public TypeInfo getOptionalDType(String name, boolean extended){
		Object value = getOptionalObject(name);

		if(value == null){
			return null;
		}

		return getDType(name, extended);
	}

	public List<TypeInfo> getDTypeList(String name, boolean extended){
		List<?> values = getList(name);

		CastFunction<TypeInfo> castFunction = new AttributeCastFunction<TypeInfo>(TypeInfo.class){

			@Override
			public TypeInfo apply(Object object){
				object = toTypeInfo(object, extended);

				return super.apply(object);
			}

			@Override
			protected String formatMessage(Object object){
				return "List attribute \'" + ClassDictUtil.formatMember(Transformer.this, name) + "\' contains an unsupported value (" + ClassDictUtil.formatClass(object) + ")";
			}
		};

		return Lists.transform(values, castFunction);
	}

	public String createFieldName(String function, Object... args){
		return createFieldName(function, Arrays.asList(args));
	}

	public String createFieldName(String function, List<?> args){
		String pmmlName = getPMMLName();

		if(pmmlName != null){
			return pmmlName;
		}

		return FieldNameUtil.create(function, args);
	}

	@Override
	public Transformer setPMMLName(String pmmlName){
		return (Transformer)super.setPMMLName(pmmlName);
	}

	static
	private Object toTypeInfo(Object object, boolean extended){

		if(object instanceof TypeConstructor){
			TypeConstructor typeConstructor = (TypeConstructor)object;

			return (TypeInfo)typeConstructor.construct();
		} else

		if(object instanceof String){
			String string = (String)object;

			if(extended){
				return new TypeInfo(){

					@Override
					public DataType getDataType(){
						return PythonTypeUtil.parseDataType(string);
					}
				};
			}
		}

		return object;
	}
}