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
package sklearn.preprocessing;

import java.util.ArrayList;
import java.util.List;

import com.google.common.collect.ContiguousSet;
import com.google.common.collect.DiscreteDomain;
import com.google.common.collect.Range;
import org.dmg.pmml.DataType;
import org.dmg.pmml.Expression;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.NormDiscrete;
import org.dmg.pmml.OpType;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.sklearn.ClassDictUtil;
import sklearn.ComplexTransformer;

public class OneHotEncoder extends ComplexTransformer {

	public OneHotEncoder(String module, String name){
		super(module, name);
	}

	@Override
	public DataType getDataType(){
		return DataType.INTEGER;
	}

	@Override
	public OpType getOpType(){
		return OpType.CATEGORICAL;
	}

	@Override
	public List<?> getClasses(){
		return getValues();
	}

	@Override
	public int getNumberOfFeatures(){
		List<? extends Number> values = getValues();

		return values.size();
	}

	@Override
	public Expression encode(int index, FieldName name){
		List<? extends Number> values = getValues();

		Number value = values.get(index);

		NormDiscrete normDicrete = new NormDiscrete(name, PMMLUtil.formatValue(value));

		return normDicrete;
	}

	public List<? extends Number> getValues(){
		List<? extends Number> featureSizes = getFeatureSizes();

		if(featureSizes.size() != 1){
			throw new IllegalArgumentException();
		}

		Object numberOfValues = get("n_values");

		if(("auto").equals(numberOfValues)){
			return getActiveFeatures();
		}

		Number featureSize = featureSizes.get(0);

		List<Integer> result = new ArrayList<>();
		result.addAll(ContiguousSet.create(Range.closedOpen(0, featureSize.intValue()), DiscreteDomain.integers()));

		return result;
	}

	public List<? extends Number> getActiveFeatures(){
		return (List)ClassDictUtil.getArray(this, "active_features_");
	}

	public List<? extends Number> getFeatureSizes(){
		return (List)ClassDictUtil.getArray(this, "n_values_");
	}
}