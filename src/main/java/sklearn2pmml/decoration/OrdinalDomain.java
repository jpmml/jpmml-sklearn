/*
 * Copyright (c) 2020 Villu Ruusmann
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

import java.util.List;

import org.dmg.pmml.DataField;
import org.dmg.pmml.OpType;
import org.jpmml.converter.ObjectFeature;
import org.jpmml.converter.PMMLEncoder;
import org.jpmml.converter.WildcardFeature;

public class OrdinalDomain extends DiscreteDomain {

	public OrdinalDomain(String module, String name){
		super(module, name);
	}

	@Override
	public OpType getOpType(){
		return OpType.ORDINAL;
	}

	@Override
	public ObjectFeature encode(WildcardFeature wildcardFeature, List<?> values){
		PMMLEncoder encoder = wildcardFeature.getEncoder();

		DataField dataField;

		if(values == null || values.isEmpty()){
			dataField = (DataField)encoder.getField(wildcardFeature.getName());
		} else

		{
			dataField = (DataField)encoder.toCategorical(wildcardFeature.getName(), standardizeValues(wildcardFeature.getDataType(), values));
		}

		dataField.setOpType(OpType.ORDINAL);

		return new ObjectFeature(encoder, dataField.getName(), dataField.getDataType());
	}
}