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

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import org.dmg.pmml.DataField;
import org.dmg.pmml.DataType;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.Expression;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.MiningField;
import org.dmg.pmml.MiningSchema;
import org.dmg.pmml.Model;
import org.dmg.pmml.OpType;
import org.dmg.pmml.PMML;
import org.dmg.pmml.mining.MiningModel;
import org.jpmml.converter.Feature;
import org.jpmml.converter.ModelEncoder;
import org.jpmml.converter.Schema;
import org.jpmml.converter.WildcardFeature;
import org.jpmml.converter.mining.MiningModelUtil;
import sklearn.Transformer;

public class SkLearnEncoder extends ModelEncoder {

	private List<Model> transformers = new ArrayList<>();


	@Override
	public PMML encodePMML(Model model){

		if(this.transformers.size() > 0){
			List<Model> models = new ArrayList<>(this.transformers);
			models.add(model);

			MiningSchema miningSchema = createMiningSchema(models);

			Schema schema = new Schema(null, Collections.emptyList());

			MiningModel miningModel = MiningModelUtil.createModelChain(models, schema)
				.setMiningSchema(miningSchema);

			model = miningModel;
		}

		return super.encodePMML(model);
	}

	public void updateFeatures(List<Feature> features, Transformer transformer){
		OpType opType;
		DataType dataType;

		try {
			opType = transformer.getOpType();
			dataType = transformer.getDataType();
		} catch(UnsupportedOperationException uoe){
			return;
		}

		for(Feature feature : features){

			if(feature instanceof WildcardFeature){
				WildcardFeature wildcardFeature = (WildcardFeature)feature;

				updateType(wildcardFeature.getName(), opType, dataType);
			}
		}
	}

	public void updateType(FieldName name, OpType opType, DataType dataType){
		DataField dataField = getDataField(name);

		if(dataField == null){
			throw new IllegalArgumentException(name.getValue());
		}

		dataField.setOpType(opType);
		dataField.setDataType(dataType);
	}

	public DataField createDataField(FieldName name){
		return createDataField(name, OpType.CONTINUOUS, DataType.DOUBLE);
	}

	public DerivedField createDerivedField(FieldName name, Expression expression){
		return createDerivedField(name, OpType.CONTINUOUS, DataType.DOUBLE, expression);
	}

	public void renameFeature(Feature feature, FieldName renamedName){
		FieldName name = feature.getName();

		DerivedField derivedField = removeDerivedField(name);

		try {
			Field field = Feature.class.getDeclaredField("name");

			if(!field.isAccessible()){
				field.setAccessible(true);
			}

			field.set(feature, renamedName);
		} catch(ReflectiveOperationException roe){
			throw new RuntimeException(roe);
		}

		derivedField.setName(renamedName);

		addDerivedField(derivedField);
	}

	public void addTransformer(Model transformer){
		this.transformers.add(transformer);
	}

	static
	private MiningSchema createMiningSchema(List<Model> models){
		List<MiningField> targetMiningFields = new ArrayList<>();

		for(Model model : models){
			MiningSchema miningSchema = model.getMiningSchema();

			List<MiningField> miningFields = miningSchema.getMiningFields();
			for(MiningField miningField : miningFields){
				MiningField.UsageType usageType = miningField.getUsageType();

				switch(usageType){
					case PREDICTED:
					case TARGET:
						targetMiningFields.add(miningField);
						break;
					default:
						break;
				}
			}
		}

		MiningSchema miningSchema = new MiningSchema(targetMiningFields);

		return miningSchema;
	}
}