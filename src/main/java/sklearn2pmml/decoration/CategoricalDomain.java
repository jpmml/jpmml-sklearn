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
package sklearn2pmml.decoration;

import java.util.Collections;
import java.util.List;
import java.util.Map;

import com.google.common.base.Function;
import com.google.common.collect.Lists;
import org.dmg.pmml.DataType;
import org.dmg.pmml.DiscrStats;
import org.dmg.pmml.OpType;
import org.dmg.pmml.UnivariateStats;
import org.jpmml.converter.Feature;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.converter.ValueUtil;
import org.jpmml.converter.WildcardFeature;
import org.jpmml.sklearn.ClassDictUtil;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn.TypeUtil;

public class CategoricalDomain extends Domain {

	public CategoricalDomain(String module, String name){
		super(module, name);
	}

	@Override
	public OpType getOpType(){
		return OpType.CATEGORICAL;
	}

	@Override
	public DataType getDataType(){
		Boolean withData = getWithData();

		if(withData){
			List<?> data = getData();

			return TypeUtil.getDataType(data, DataType.STRING);
		}

		return DataType.STRING;
	}

	@Override
	public List<Feature> encodeFeatures(List<Feature> features, SkLearnEncoder encoder){
		Boolean withData = getWithData();
		Boolean withStatistics = getWithStatistics();

		ClassDictUtil.checkSize(1, features);

		Feature feature = features.get(0);

		WildcardFeature wildcardFeature = (WildcardFeature)feature;

		if(withData){
			List<?> data = getData();

			Function<Object, String> function = new Function<Object, String>(){

				@Override
				public String apply(Object object){
					return ValueUtil.formatValue(object);
				}
			};

			List<String> categories = Lists.transform(data, function);

			feature = wildcardFeature.toCategoricalFeature(categories);
		} // End if

		if(withStatistics){
			Map<String, ?> counts = extractMap(getCounts(), 0);
			Object[] discrStats = getDiscrStats();

			UnivariateStats univariateStats = new UnivariateStats()
				.setField(wildcardFeature.getName())
				.setCounts(createCounts(counts))
				.setDiscrStats(createDiscrStats(discrStats));

			encoder.putUnivariateStats(univariateStats);
		}

		return super.encodeFeatures(Collections.<Feature>singletonList(feature), encoder);
	}

	public List<?> getData(){
		return (List)ClassDictUtil.getArray(this, "data_");
	}

	public Object[] getDiscrStats(){
		return (Object[])get("discr_stats_");
	}

	static
	public DiscrStats createDiscrStats(Object[] objects){
		List<Object> values = (List)asArray(objects[0]);
		List<Integer> counts = ValueUtil.asIntegers((List)asArray(objects[1]));

		ClassDictUtil.checkSize(values, counts);

		DiscrStats discrStats = new DiscrStats()
			.addArrays(PMMLUtil.createStringArray(values), PMMLUtil.createIntArray(counts));

		return discrStats;
	}
}