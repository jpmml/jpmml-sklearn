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
import java.util.Collections;
import java.util.List;
import java.util.Set;

import org.dmg.pmml.DataType;
import org.dmg.pmml.DefineFunction;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.Model;
import org.dmg.pmml.OpType;
import org.jpmml.converter.Feature;
import org.jpmml.converter.Schema;
import org.jpmml.converter.ValueUtil;
import org.jpmml.sklearn.FeatureMapper;

abstract
public class Estimator extends BaseEstimator implements HasNumberOfFeatures {

	public Estimator(String module, String name){
		super(module, name);
	}

	abstract
	public Schema createSchema(FeatureMapper featureMapper);

	abstract
	public Model encodeModel(Schema schema);

	public int getNumberOfFeatures(){
		return ValueUtil.asInt((Number)get("n_features_"));
	}

	public boolean requiresContinuousInput(){
		return true;
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

	public Set<DefineFunction> encodeDefineFunctions(){
		return Collections.emptySet();
	}

	public boolean validateSchema(Schema schema){
		int numberOfFeatures = getNumberOfFeatures();

		List<Feature> features = schema.getFeatures();

		return (features.size() == numberOfFeatures);
	}

	protected FieldName createTargetField(){
		return FieldName.create("y");
	}

	protected List<FieldName> createActiveFields(int size){
		List<FieldName> result = new ArrayList<>(size);

		for(int i = 0; i < size; i++){
			result.add(FieldName.create("x" + String.valueOf(i + 1)));
		}

		return result;
	}
}