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
package category_encoders;

import java.util.Arrays;
import java.util.List;

import org.dmg.pmml.DataType;
import org.dmg.pmml.OpType;
import org.dmg.pmml.PMML;
import org.jpmml.sklearn.Encodable;
import sklearn.HasFeatureNamesIn;
import sklearn.Transformer;
import sklearn.TransformerUtil;

abstract
public class BaseEncoder extends Transformer implements HasFeatureNamesIn, Encodable, BaseEncoderConstants {

	public BaseEncoder(String module, String name){
		super(module, name);
	}

	@Override
	public OpType getOpType(){
		return OpType.CATEGORICAL;
	}

	@Override
	public DataType getDataType(){
		return DataType.STRING;
	}

	@Override
	public PMML encodePMML(){
		return TransformerUtil.encodePMML(this);
	}

	public List<Object> getCols(){
		return getObjectList("cols");
	}

	public List<String> getInvariantCols(){

		// CategoryEncoders 2.3
		if(hasattr("drop_cols")){
			return getStringList("drop_cols");
		}

		// CategoryEncoders 2.5+
		return getStringList("invariant_cols");
	}

	public Boolean getDropInvariant(){
		return getEnum("drop_invariant", this::getBoolean, Arrays.asList(Boolean.FALSE));
	}

	public List<String> getFeatureNamesOut(){

		// CategoryEncoders 2.5.1post0
		if(hasattr("feature_names")){
			return getStringList("feature_names");
		}

		// CategoryEncoders 2.6+
		return getStringList("feature_names_out_");
	}

	public String getHandleMissing(){
		return getEnum("handle_missing", this::getString, BaseEncoder.ENUM_HANDLEMISSING);
	}

	public String getHandleUnknown(){
		return getEnum("handle_unknown", this::getString, BaseEncoder.ENUM_HANDLEUNKNOWN);
	}

	public static final Object CATEGORY_NAN = Double.NaN;
}