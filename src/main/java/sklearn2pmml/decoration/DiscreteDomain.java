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
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn.StepUtil;

abstract
public class DiscreteDomain extends Domain {

	public DiscreteDomain(String module, String name){
		super(module, name);
	}

	abstract
	public Feature encodeFeature(WildcardFeature wildcardFeature, List<?> values);

	@Override
	public int getNumberOfFeatures(){
		return 1;
	}

	@Override
	public DataType getDataType(){
		Object dtype = getDType();
		Boolean withData = getWithData();

		if(dtype != null){
			return StepUtil.getDataType(dtype);
		} // End if

		if(withData){
			List<?> data = getData();

			return TypeUtil.getDataType(data, DataType.STRING);
		}

		return DataType.STRING;
	}

	@Override
	public List<Feature> encodeFeatures(List<Feature> features, SkLearnEncoder encoder){
		features = super.encodeFeatures(features, encoder);

		Boolean withData = getWithData();
		Boolean withStatistics = getWithStatistics();

		ClassDictUtil.checkSize(1, features);

		Feature feature = features.get(0);

		WildcardFeature wildcardFeature = asWildcardFeature(feature);

		if(withData){
			List<?> data = getData();

			feature = encodeFeature(wildcardFeature, data);
		} else

		{
			feature = encodeFeature(wildcardFeature, Collections.emptyList());
		} // End if

		if(withStatistics){
			Map<String, ?> counts = extractMap(getCounts(), 0);
			Object[] discrStats = getDiscrStats();

			UnivariateStats univariateStats = new UnivariateStats()
				.setField(wildcardFeature.getName())
				.setCounts(createCounts(counts))
				.setDiscrStats(createDiscrStats(wildcardFeature.getDataType(), discrStats));

			encoder.putUnivariateStats(univariateStats);
		}

		return Collections.singletonList(feature);
	}

	public List<?> getData(){
		return getArray("data_");
	}

	public Object[] getDiscrStats(){
		return getTuple("discr_stats_");
	}

	static
	public DiscrStats createDiscrStats(DataType dataType, Object[] objects){
		List<Object> values = (List)asArray(objects[0]);
		List<Integer> counts = ValueUtil.asIntegers((List)asArray(objects[1]));

		ClassDictUtil.checkSize(values, counts);

		DiscrStats discrStats = new DiscrStats()
			.addArrays(PMMLUtil.createStringArray(standardizeValues(dataType, values)), PMMLUtil.createIntArray(counts));

		return discrStats;
	}
}