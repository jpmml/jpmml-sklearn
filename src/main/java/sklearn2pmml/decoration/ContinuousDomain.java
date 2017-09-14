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

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import org.dmg.pmml.Interval;
import org.dmg.pmml.NumericInfo;
import org.dmg.pmml.OpType;
import org.dmg.pmml.UnivariateStats;
import org.jpmml.converter.Feature;
import org.jpmml.converter.ValidValueDecorator;
import org.jpmml.converter.ValueUtil;
import org.jpmml.converter.WildcardFeature;
import org.jpmml.sklearn.ClassDictUtil;
import org.jpmml.sklearn.SkLearnEncoder;

public class ContinuousDomain extends Domain {

	public ContinuousDomain(String module, String name){
		super(module, name);
	}

	@Override
	public OpType getOpType(){
		return OpType.CONTINUOUS;
	}

	@Override
	public List<Feature> encodeFeatures(List<Feature> features, SkLearnEncoder encoder){
		Boolean withData = getWithData();
		Boolean withStatistics = getWithStatistics();

		List<? extends Number> dataMin = null;
		List<? extends Number> dataMax = null;

		if(withData){
			dataMin = getDataMin();
			dataMax = getDataMax();

			ClassDictUtil.checkSize(features, dataMin, dataMax);
		}

		List<Feature> result = new ArrayList<>();

		for(int i = 0; i < features.size(); i++){
			Feature feature = features.get(i);

			WildcardFeature wildcardFeature = (WildcardFeature)feature;

			if(withData){
				Interval interval = new Interval(Interval.Closure.CLOSED_CLOSED)
					.setLeftMargin(ValueUtil.asDouble(dataMin.get(i)))
					.setRightMargin(ValueUtil.asDouble(dataMax.get(i)));

				ValidValueDecorator validValueDecorator = new ValidValueDecorator()
					.addIntervals(interval);

				feature = wildcardFeature.toContinuousFeature();

				encoder.addDecorator(wildcardFeature.getName(), validValueDecorator);
			} // End if

			if(withStatistics){
				Map<String, ?> counts = extractMap(getCounts(), i);
				Map<String, ?> numericInfo = extractMap(getNumericInfo(), i);

				UnivariateStats univariateStats = new UnivariateStats()
					.setField(wildcardFeature.getName())
					.setCounts(createCounts(counts))
					.setNumericInfo(createNumericInfo(numericInfo));

				encoder.putUnivariateStats(univariateStats);
			}

			result.add(feature);
		}

		return super.encodeFeatures(result, encoder);
	}

	public List<? extends Number> getDataMin(){
		return (List)ClassDictUtil.getArray(this, "data_min_");
	}

	public List<? extends Number> getDataMax(){
		return (List)ClassDictUtil.getArray(this, "data_max_");
	}

	public Map<String, ?> getNumericInfo(){
		return (Map)get("numeric_info_");
	}

	static
	public NumericInfo createNumericInfo(Map<String, ?> values){
		NumericInfo numericInfo = new NumericInfo()
			.setMinimum(selectValue(values, "minimum"))
			.setMaximum(selectValue(values, "maximum"))
			.setMean(selectValue(values, "mean"))
			.setStandardDeviation(selectValue(values, "standardDeviation"))
			.setMedian(selectValue(values, "median"))
			.setInterQuartileRange(selectValue(values, "interQuartileRange"));

		return numericInfo;
	}
}