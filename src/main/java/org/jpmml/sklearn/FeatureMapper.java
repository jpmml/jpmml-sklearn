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

import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
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

	private List<List<Feature>> rows = new ArrayList<>();


	public Schema createSupervisedSchema(){
		Feature feature = getTargetFeature();

		DataField dataField = getDataField(feature.getName());
		if(dataField == null){
			throw new IllegalArgumentException();
		}

		FieldName targetField = dataField.getName();

		List<String> targetCategories = null;

		if(dataField.hasValues()){
			targetCategories = PMMLUtil.getValues(dataField);
		}

		List<FieldName> activeFields = new ArrayList<>(getDataFields().keySet());
		activeFields.remove(targetField);

		List<Feature> features = getActiveFeatures(true);

		Schema schema = new Schema(targetField, targetCategories, activeFields, features);

		return schema;
	}

	public Schema createUnsupervisedSchema(){
		List<FieldName> activeFields = new ArrayList<>(getDataFields().keySet());

		List<Feature> features = getActiveFeatures(false);

		Schema schema = new Schema(null, null, activeFields, features);

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

		field
			.setOpType(opType)
			.setDataType(dataType);
	}

	public void updateValueSpace(FieldName name, List<String> categories){
		TypeDefinitionField field = getField(name);

		List<Value> values = field.getValues();
		if(values.size() > 0){
			throw new IllegalArgumentException();
		} // End if

		if(categories != null && categories.size() > 0){
			values.addAll(PMMLUtil.createValues(categories));
		}
	}

	public void initTargetField(FieldName name, OpType opType, DataType dataType, List<String> targetCategories){
		DataField dataField = createDataField(name, opType, dataType);

		updateValueSpace(dataField.getName(), targetCategories);

		Feature feature = new WildcardFeature(dataField);

		addRow(Collections.singletonList(feature));
	}

	public void updateTargetField(OpType opType, DataType dataType, List<String> targetCategories){
		Feature feature = getTargetFeature();

		DataField dataField = getDataField(feature.getName());
		if(dataField == null){
			throw new IllegalArgumentException();
		}

		updateType(dataField.getName(), opType, dataType);
		updateValueSpace(dataField.getName(), targetCategories);
	}

	public void initActiveFields(List<FieldName> names, OpType opType, DataType dataType){

		if(!(OpType.CONTINUOUS).equals(opType)){
			throw new IllegalArgumentException();
		}

		for(FieldName name : names){
			DataField dataField = createDataField(name, opType, dataType);

			Feature feature = new ContinuousFeature(dataField);

			addRow(Collections.singletonList(feature));
		}
	}

	public void updateActiveFields(boolean supervised, OpType opType, DataType dataType){

		if(!(OpType.CONTINUOUS).equals(opType)){
			throw new IllegalArgumentException();
		}

		int count = 0;

		List<List<Feature>> activeRows = getActiveRows(supervised);
		for(List<Feature> activeRow : activeRows){

			for(ListIterator<Feature> featureIt = activeRow.listIterator(); featureIt.hasNext(); ){
				Feature feature = featureIt.next();

				count++;

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

	public WildcardFeature getTargetFeature(){
		List<Feature> targetRow = getTargetRow();
		if(targetRow.size() != 1){
			throw new IllegalArgumentException();
		}

		WildcardFeature feature = (WildcardFeature)targetRow.get(0);

		return feature;
	}

	public List<Feature> getActiveFeatures(boolean supervised){
		List<List<Feature>> activeRows = getActiveRows(supervised);

		List<Feature> features = Lists.newArrayList(Iterables.concat(activeRows));

		return features;
	}

	public void addRow(List<Feature> row){
		this.rows.add(row);
	}

	public boolean isEmpty(){
		return this.rows.isEmpty();
	}

	public List<List<Feature>> getRows(){
		return this.rows;
	}

	public List<Feature> getTargetRow(){
		return this.rows.get(this.rows.size() - 1);
	}

	public List<List<Feature>> getActiveRows(boolean supervised){

		if(supervised){
			return this.rows.subList(0, this.rows.size() - 1);
		}

		return this.rows;
	}
}