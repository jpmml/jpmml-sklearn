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

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import org.dmg.pmml.DataType;
import org.dmg.pmml.DiscrStats;
import org.dmg.pmml.UnivariateStats;
import org.jpmml.converter.Feature;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.converter.TypeUtil;
import org.jpmml.converter.ValueUtil;
import org.jpmml.converter.WildcardFeature;
import org.jpmml.python.ClassDictUtil;
import org.jpmml.python.TypeInfo;
import org.jpmml.sklearn.SkLearnEncoder;
import pandas.core.CategoricalDtype;
import sklearn.HasMultiType;

abstract
public class DiscreteDomain extends Domain implements HasMultiType {

	public DiscreteDomain(String module, String name){
		super(module, name);
	}

	abstract
	public Feature encodeFeature(WildcardFeature wildcardFeature, List<?> values);

	@Override
	public DataType getDataType(){
		TypeInfo dtype = getDType();

		if(dtype != null){
			return dtype.getDataType();
		}

		return DataType.STRING;
	}

	@Override
	public DataType getDataType(int index){
		Boolean withData = getWithData();

		if(withData){
			List<? extends List<?>> dataValues = getDataValues();

			List<?> values = dataValues.get(index);

			return TypeUtil.getDataType(values, DataType.STRING);
		}

		return getDataType();
	}

	@Override
	public List<Feature> encodeFeatures(List<Feature> features, SkLearnEncoder encoder){
		features = super.encodeFeatures(features, encoder);

		TypeInfo dtype = getDType();
		Boolean withData = getWithData();
		Boolean withStatistics = getWithStatistics();

		List<? extends List<?>> dataValues = null;

		if(withData){
			dataValues = getDataValues();

			ClassDictUtil.checkSize(features.size(), dataValues);
		}

		Map<String, ?> counts = null;
		List<Object[]> discrStats = null;

		if(withStatistics){
			counts = getCounts();
			discrStats = getDiscrStats();

			ClassDictUtil.checkSize(features.size(), discrStats);
		}

		List<Feature> result = new ArrayList<>();

		for(int i = 0; i < features.size(); i++){
			Feature feature = features.get(i);

			WildcardFeature wildcardFeature = asWildcardFeature(feature);

			List<?> values = Collections.emptyList();

			if(withData){
				values = dataValues.get(i);
			} else

			{
				if(dtype instanceof CategoricalDtype){
					CategoricalDtype categoricalDtype = (CategoricalDtype)dtype;

					values = categoricalDtype.getValues();
				}
			}

			feature = encodeFeature(wildcardFeature, values);

			if(withStatistics){
				UnivariateStats univariateStats = new UnivariateStats()
					.setField(wildcardFeature.getName())
					.setCounts(createCounts(extractMap(counts, i)))
					.setDiscrStats(createDiscrStats(wildcardFeature.getDataType(), discrStats.get(i)));

				encoder.addUnivariateStats(univariateStats);
			}

			result.add(feature);
		}

		return result;
	}

	public List<List<Object>> getDataValues(){

		// SkLearn2PMML 0.101.0+
		if(hasattr("data_values_")){
			Object dataValues = getObject("data_values_");

			if(dataValues instanceof List){
				return getArrayList("data_values_", Object.class);
			}

			return Collections.singletonList(getObjectArray("data_values_"));
		} else

		// SkLearn2PMML 0.100.2
		{
			return Collections.singletonList(getObjectArray("data_"));
		}
	}

	public List<Object[]> getDiscrStats(){
		Object discrStats = getObject("discr_stats_");

		if(discrStats instanceof List){
			return getTupleList("discr_stats_");
		}

		return Collections.singletonList(getTuple("discr_stats_"));
	}

	@SuppressWarnings({"rawtypes", "unchecked"})
	static
	public DiscrStats createDiscrStats(DataType dataType, Object[] objects){
		List<?> values = asArray(objects[0]);
		List<Integer> counts = ValueUtil.asIntegers((List)asArray(objects[1]));

		ClassDictUtil.checkSize(values, counts);

		DiscrStats discrStats = new DiscrStats()
			.addArrays(PMMLUtil.createStringArray(standardizeValues(dataType, values)), PMMLUtil.createIntArray(counts));

		return discrStats;
	}
}