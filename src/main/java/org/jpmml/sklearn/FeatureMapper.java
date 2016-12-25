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
package org.jpmml.sklearn;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.ListIterator;

import com.google.common.base.Function;
import org.dmg.pmml.DataField;
import org.dmg.pmml.DataType;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.Expression;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.NormDiscrete;
import org.dmg.pmml.OpType;
import org.dmg.pmml.TypeDefinitionField;
import org.dmg.pmml.Value;
import org.jpmml.converter.BinaryFeature;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.Feature;
import org.jpmml.converter.ListFeature;
import org.jpmml.converter.PMMLMapper;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.converter.Schema;
import org.jpmml.converter.WildcardFeature;

public class FeatureMapper extends PMMLMapper {

	private List<Row> rows = new ArrayList<>();


	public Schema createSchema(FieldName targetField, List<String> targetCategories){
		List<FieldName> activeFields = new ArrayList<>(getDataFields().keySet());

		if(targetField != null){
			activeFields.remove(targetField);
		}

		Schema schema = new Schema(targetField, targetCategories, activeFields, getFeatures());

		return schema;
	}

	public Schema cast(OpType opType, DataType dataType, Schema schema){

		if(opType != null && !(OpType.CONTINUOUS).equals(opType)){
			throw new IllegalArgumentException();
		}

		String function = (dataType.name()).toLowerCase();

		List<Feature> castFeatures = new ArrayList<>();

		List<Feature> features = schema.getFeatures();
		for(Feature feature : features){

			cast:
			if(feature instanceof BinaryFeature){
				BinaryFeature binaryFeature = (BinaryFeature)feature;

				if(opType == null){
					break cast;
				}

				FieldName name = FieldName.create(function + "(" + (binaryFeature.getName()).getValue() + "=" + binaryFeature.getValue() + ")");

				DerivedField derivedField = getDerivedField(name);
				if(derivedField == null){
					NormDiscrete normDiscrete = new NormDiscrete(binaryFeature.getName(), binaryFeature.getValue());

					derivedField = createDerivedField(name, OpType.CONTINUOUS, dataType, normDiscrete);
				}

				feature = new ContinuousFeature(derivedField);
			} else

			if(feature instanceof ContinuousFeature){
				ContinuousFeature continuousFeature = (ContinuousFeature)feature;

				if((continuousFeature instanceof ListFeature) || (continuousFeature.getDataType()).equals(dataType)){
					break cast;
				}

				FieldName name = FieldName.create(function + "(" + (continuousFeature.getName()).getValue() + ")");

				DerivedField derivedField = getDerivedField(name);
				if(derivedField == null){
					derivedField = createDerivedField(name, OpType.CONTINUOUS, dataType, continuousFeature.ref());
				}

				feature = new ContinuousFeature(derivedField);
			}

			castFeatures.add(feature);
		}

		Schema castSchema = new Schema(schema.getTargetField(), schema.getTargetCategories(), schema.getActiveFields(), castFeatures);

		return castSchema;
	}

	public void updateType(FieldName name, OpType opType, DataType dataType){
		TypeDefinitionField field = getField(name);

		updateType(field, opType, dataType);
	}

	public void updateType(TypeDefinitionField field, OpType opType, DataType dataType){
		field
			.setOpType(opType)
			.setDataType(dataType);
	}

	public void updateValueSpace(FieldName name, List<String> categories){
		TypeDefinitionField field = getField(name);

		updateValueSpace(field, categories);
	}

	public void updateValueSpace(TypeDefinitionField field, List<String> categories){
		DataField dataField = (DataField)field;

		List<Value> values = dataField.getValues();
		if(values.size() > 0){
			throw new IllegalArgumentException();
		} // End if

		if(categories != null && categories.size() > 0){
			values.addAll(PMMLUtil.createValues(categories));
		}
	}

	public void initFeatures(List<FieldName> names, OpType opType, DataType dataType){

		if(!(OpType.CONTINUOUS).equals(opType)){
			throw new IllegalArgumentException();
		}

		for(FieldName name : names){
			String id = name.getValue();

			DataField dataField = createDataField(name, opType, dataType);

			Feature feature = new ContinuousFeature(dataField);

			addRow(Collections.singletonList(id), Collections.singletonList(feature));
		}
	}

	public void updateFeatures(OpType opType, DataType dataType){

		if(!(OpType.CONTINUOUS).equals(opType)){
			throw new IllegalArgumentException();
		}

		List<Row> rows = this.rows;
		for(Row row : rows){
			List<Feature> features = row.getFeatures();

			for(ListIterator<Feature> featureIt = features.listIterator(); featureIt.hasNext(); ){
				Feature feature = featureIt.next();

				if(feature instanceof BinaryFeature){
					continue;
				}

				updateType(feature.getName(), opType, dataType);

				if(feature instanceof WildcardFeature){
					DataField dataField = (DataField)getField(feature.getName());

					feature = new ContinuousFeature(dataField);

					featureIt.set(feature);
				}
			}
		}
	}

	public DataField createDataField(FieldName name){
		return createDataField(name, OpType.CONTINUOUS, DataType.DOUBLE);
	}

	public DerivedField createDerivedField(FieldName name, Expression expression){
		return createDerivedField(name, OpType.CONTINUOUS, DataType.DOUBLE, expression);
	}

	public List<String> getIds(){
		Function<Row, List<String>> function = new Function<Row, List<String>>(){

			@Override
			public List<String> apply(Row row){
				return row.getIds();
			}
		};

		return getRowData(function);
	}

	public List<Feature> getFeatures(){
		Function<Row, List<Feature>> function = new Function<Row, List<Feature>>(){

			@Override
			public List<Feature> apply(Row row){
				return row.getFeatures();
			}
		};

		return getRowData(function);
	}

	public void addRow(List<String> ids, List<Feature> features){
		Row row = new Row(ids, features);

		this.rows.add(row);
	}

	private <E> List<E> getRowData(Function<Row, List<E>> function){
		List<E> result = new ArrayList<>();

		List<Row> rows = this.rows;
		for(Row row : rows){
			result.addAll(function.apply(row));
		}

		return result;
	}

	static
	private class Row {

		private List<String> ids = null;

		private List<Feature> features = null;


		private Row(List<String> ids, List<Feature> features){
			setIds(ids);
			setFeatures(features);
		}

		public List<String> getIds(){
			return this.ids;
		}

		private void setIds(List<String> ids){
			this.ids = new ArrayList<>(ids);
		}

		public List<Feature> getFeatures(){
			return this.features;
		}

		private void setFeatures(List<Feature> features){
			this.features = new ArrayList<>(features);
		}
	}
}