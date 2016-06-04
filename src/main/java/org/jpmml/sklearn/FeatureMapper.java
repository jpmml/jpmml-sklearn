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
import java.util.Collection;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.ListIterator;
import java.util.Map;

import com.google.common.base.Function;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import org.dmg.pmml.DataDictionary;
import org.dmg.pmml.DataField;
import org.dmg.pmml.DataType;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.Expression;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.NormDiscrete;
import org.dmg.pmml.OpType;
import org.dmg.pmml.PMML;
import org.dmg.pmml.TransformationDictionary;
import org.dmg.pmml.TypeDefinitionField;
import org.dmg.pmml.Value;
import org.jpmml.converter.BinaryFeature;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.Feature;
import org.jpmml.converter.FeatureSchema;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.converter.PseudoFeature;

public class FeatureMapper {

	private List<List<Feature>> steps = new ArrayList<>();

	private Map<FieldName, DataField> dataFields = new LinkedHashMap<>();

	private Map<FieldName, DerivedField> derivedFields = new LinkedHashMap<>();


	public PMML encodePMML(){

		if(!Collections.disjoint(this.dataFields.keySet(), this.derivedFields.keySet())){
			throw new IllegalArgumentException();
		}

		List<DataField> dataFields = new ArrayList<>(this.dataFields.values());
		List<DerivedField> derivedFields = new ArrayList<>(this.derivedFields.values());

		DataDictionary dataDictionary = new DataDictionary();
		(dataDictionary.getDataFields()).addAll(dataFields);

		TransformationDictionary transformationDictionary = null;
		if(derivedFields.size() > 0){
			transformationDictionary = new TransformationDictionary();
			(transformationDictionary.getDerivedFields()).addAll(derivedFields);
		}

		PMML pmml = new PMML("4.2", PMMLUtil.createHeader("JPMML-SkLearn", "1.0-SNAPSHOT"), dataDictionary)
			.setTransformationDictionary(transformationDictionary);

		return pmml;
	}

	public FeatureSchema createSupervisedSchema(){
		PseudoFeature feature = getTargetFeature();

		DataField dataField = this.dataFields.get(feature.getName());
		if(dataField == null){
			throw new IllegalArgumentException();
		}

		FieldName targetField = dataField.getName();

		List<String> targetCategories = null;

		if(dataField.hasValues()){
			List<Value> values = dataField.getValues();

			Function<Value, String> function = new Function<Value, String>(){

				@Override
				public String apply(Value value){
					return value.getValue();
				}
			};

			targetCategories = new ArrayList<>(Lists.transform(values, function));
		}

		List<FieldName> activeFields = new ArrayList<>(this.dataFields.keySet());
		activeFields.remove(targetField);

		List<Feature> features = flatten(getActiveSteps(true));

		FeatureSchema schema = new FeatureSchema(targetField, targetCategories, activeFields, features);

		return schema;
	}

	public FeatureSchema createUnsupervisedSchema(){
		List<FieldName> activeFields = new ArrayList<>(this.dataFields.keySet());

		List<Feature> features = flatten(getActiveSteps(false));

		FeatureSchema schema = new FeatureSchema(null, null, activeFields, features);

		return schema;
	}

	public TypeDefinitionField getField(FieldName name){
		DataField dataField = this.dataFields.get(name);
		DerivedField derivedField = this.derivedFields.get(name);

		if(dataField != null && derivedField == null){
			return dataField;
		} else

		if(dataField == null && derivedField != null){
			return derivedField;
		}

		throw new IllegalArgumentException();
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

		Feature feature = new PseudoFeature(dataField);

		addStep(Collections.singletonList(feature));
	}

	public void updateTargetField(OpType opType, DataType dataType, List<String> targetCategories){
		PseudoFeature feature = getTargetFeature();

		DataField dataField = this.dataFields.get(feature.getName());
		if(dataField == null){
			throw new IllegalArgumentException();
		}

		updateType(dataField.getName(), opType, dataType);
		updateValueSpace(dataField.getName(), targetCategories);
	}

	public void initActiveFields(List<FieldName> names, OpType opType, DataType dataType){

		for(FieldName name : names){
			DataField dataField = createDataField(name, opType, dataType);

			Feature feature = createActiveFeature(dataField);

			addStep(Collections.singletonList(feature));
		}
	}

	public void simplifyActiveFields(boolean supervised, OpType opType, DataType dataType){
		List<List<Feature>> activeSteps = getActiveSteps(supervised);

		for(List<Feature> activeStep : activeSteps){

			for(ListIterator<Feature> featureIt = activeStep.listIterator(); featureIt.hasNext(); ){
				Feature feature = featureIt.next();

				if(feature instanceof BinaryFeature){
					BinaryFeature binaryFeature = (BinaryFeature)feature;

					NormDiscrete normDiscrete = new NormDiscrete(binaryFeature.getName(), binaryFeature.getValue());

					DerivedField derivedField = createDerivedField(FieldName.create((binaryFeature.getName()).getValue() + "=" + binaryFeature.getValue()), opType, dataType, normDiscrete);

					feature = new ContinuousFeature(derivedField);

					featureIt.set(feature);
				}
			}
		}
	}

	public void updateActiveFields(int numberOfFeatures, boolean supervised, OpType opType, DataType dataType){
		List<List<Feature>> activeSteps = getActiveSteps(supervised);

		int count = 0;

		for(List<Feature> activeStep : activeSteps){

			for(ListIterator<Feature> featureIt = activeStep.listIterator(); featureIt.hasNext(); ){
				Feature feature = featureIt.next();

				count++;

				if(feature instanceof BinaryFeature){
					continue;
				}

				updateType(feature.getName(), opType, dataType);

				if(feature instanceof PseudoFeature){
					DataField dataField = (DataField)getField(feature.getName());

					feature = createActiveFeature(dataField);

					featureIt.set(feature);
				}
			}
		}

		if(count != numberOfFeatures){
			throw new IllegalArgumentException();
		}
	}

	public DataField createDataField(FieldName name){
		return createDataField(name, OpType.CONTINUOUS, DataType.DOUBLE);
	}

	public DataField createDataField(FieldName name, OpType opType, DataType dataType){
		checkName(name);

		DataField dataField = new DataField(name, opType, dataType);

		this.dataFields.put(dataField.getName(), dataField);

		return dataField;
	}

	public DerivedField createDerivedField(FieldName name, Expression expression){
		return createDerivedField(name, OpType.CONTINUOUS, DataType.DOUBLE, expression);
	}

	public DerivedField createDerivedField(FieldName name, OpType opType, DataType dataType, Expression expression){
		checkName(name);

		DerivedField derivedField = new DerivedField(opType, dataType)
			.setName(name)
			.setExpression(expression);

		this.derivedFields.put(derivedField.getName(), derivedField);

		return derivedField;
	}

	public void addStep(List<Feature> step){
		this.steps.add(step);
	}

	public boolean isEmpty(){
		return this.steps.isEmpty();
	}

	public PseudoFeature getTargetFeature(){
		List<Feature> targetStep = getTargetStep();
		if(targetStep.size() != 1){
			throw new IllegalArgumentException();
		}

		PseudoFeature feature = (PseudoFeature)targetStep.get(0);

		return feature;
	}

	public List<List<Feature>> getSteps(){
		return this.steps;
	}

	public List<Feature> getTargetStep(){
		return this.steps.get(this.steps.size() - 1);
	}

	public List<List<Feature>> getActiveSteps(boolean supervised){

		if(supervised){
			return this.steps.subList(0, this.steps.size() - 1);
		}

		return this.steps;
	}

	private void checkName(FieldName name){

		if(this.dataFields.containsKey(name) || this.derivedFields.containsKey(name)){
			throw new IllegalArgumentException();
		}
	}

	private Feature createActiveFeature(DataField dataField){
		OpType opType = dataField.getOpType();

		switch(opType){
			case CONTINUOUS:
				return new ContinuousFeature(dataField);
			default:
				return new PseudoFeature(dataField);
		}
	}

	static
	private <E> List<E> flatten(Collection<? extends Collection<E>> collections){
		List<E> result = Lists.newArrayList(Iterables.concat(collections));

		return result;
	}
}