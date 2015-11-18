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

import java.util.Collections;
import java.util.List;
import java.util.Set;

import org.dmg.pmml.DataDictionary;
import org.dmg.pmml.DataField;
import org.dmg.pmml.DataType;
import org.dmg.pmml.DefineFunction;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.Model;
import org.dmg.pmml.OpType;
import org.dmg.pmml.PMML;
import org.dmg.pmml.TransformationDictionary;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.sklearn.Schema;

abstract
public class Estimator extends BaseEstimator {

	public Estimator(String module, String name){
		super(module, name);
	}

	abstract
	public Schema createSchema();

	abstract
	public DataField encodeTargetField(FieldName name, List<String> targetCategories);

	abstract
	public Model encodeModel(Schema schema);

	public int getNumberOfFeatures(){
		return (Integer)get("n_features_");
	}

	/**
	 * The {@link OpType operational type} of active fields.
	 */
	public OpType getOpType(){
		return OpType.CONTINUOUS;
	}

	/**
	 * The {@link DataType data type} of active fields.
	 */
	public DataType getDataType(){
		return DataType.DOUBLE;
	}

	public DataDictionary encodeDataDictionary(Schema schema){
		DataDictionary dataDictionary = new DataDictionary(null);

		FieldName targetField = schema.getTargetField();
		if(targetField != null){
			DataField dataField = encodeTargetField(targetField, schema.getTargetCategories());

			dataDictionary.addDataFields(dataField);
		}

		List<FieldName> activeFields = schema.getActiveFields();
		for(FieldName activeField : activeFields){
			DataField dataField = new DataField(activeField, getOpType(), getDataType());

			dataDictionary.addDataFields(dataField);
		}

		return dataDictionary;
	}

	public Set<DefineFunction> encodeDefineFunctions(){
		return Collections.emptySet();
	}

	public TransformationDictionary encodeTransformationDictionary(Schema schema){
		Set<DefineFunction> defineFunctions = encodeDefineFunctions();

		if(defineFunctions.isEmpty()){
			return null;
		}

		TransformationDictionary transformationDictionary = new TransformationDictionary();

		for(DefineFunction defineFunction : defineFunctions){
			transformationDictionary.addDefineFunctions(defineFunction);
		}

		return transformationDictionary;
	}

	public PMML encodePMML(Schema schema){
		DataDictionary dataDictionary = encodeDataDictionary(schema);
		TransformationDictionary transformationDictionary = encodeTransformationDictionary(schema);

		PMML pmml = new PMML("4.2", PMMLUtil.createHeader("JPMML-SkLearn"), dataDictionary)
			.setTransformationDictionary(transformationDictionary);

		Model model = encodeModel(schema);

		pmml.addModels(model);

		return pmml;
	}
}