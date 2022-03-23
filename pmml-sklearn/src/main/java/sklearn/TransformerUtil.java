/*
 * Copyright (c) 2022 Villu Ruusmann
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

import org.dmg.pmml.DataType;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.Field;
import org.dmg.pmml.OpType;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.Feature;
import org.jpmml.converter.ObjectFeature;
import org.jpmml.converter.PMMLEncoder;
import org.jpmml.converter.StringFeature;
import org.jpmml.sklearn.SkLearnEncoder;

public class TransformerUtil {

	private TransformerUtil(){
	}

	static
	public Feature createFeature(Field<?> field, SkLearnEncoder encoder){
		DataType dataType = field.requireDataType();
		OpType opType = field.requireOpType();

		switch(dataType){
			case STRING:
				return new StringFeature(encoder, field);
			case INTEGER:
			case FLOAT:
			case DOUBLE:
				switch(opType){
					case CONTINUOUS:
						return new ContinuousFeature(encoder, field);
					default:
						return new ObjectFeature(encoder, field){

							@Override
							public ContinuousFeature toContinuousFeature(){
								PMMLEncoder encoder = getEncoder();

								DerivedField derivedField = (DerivedField)encoder.toContinuous(getName());

								return new ContinuousFeature(encoder, derivedField);
							}
						};
				}
			default:
				return new ObjectFeature(encoder, field);
		}
	}

	static
	public OpType getOpType(DataType dataType){

		switch(dataType){
			case STRING:
				return OpType.CATEGORICAL;
			case INTEGER:
			case FLOAT:
			case DOUBLE:
				return OpType.CONTINUOUS;
			case BOOLEAN:
				return OpType.CATEGORICAL;
			case DATE:
			case DATE_TIME:
				return OpType.ORDINAL;
			default:
				throw new IllegalArgumentException();
		}
	}

	static
	public DataType parseDataType(String dtype){

		switch(dtype){
			case "datetime64[D]":
				return DataType.DATE;
			case "datetime64[s]":
				return DataType.DATE_TIME;
			default:
				throw new IllegalArgumentException("Python data type \'" + dtype + "\' is not supported");
		}
	}
}