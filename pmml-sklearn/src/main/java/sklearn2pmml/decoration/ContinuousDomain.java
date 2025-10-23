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
import java.util.Arrays;
import java.util.List;
import java.util.Map;

import org.dmg.pmml.DataField;
import org.dmg.pmml.DataType;
import org.dmg.pmml.Interval;
import org.dmg.pmml.NumericInfo;
import org.dmg.pmml.OpType;
import org.dmg.pmml.OutlierTreatmentMethod;
import org.dmg.pmml.UnivariateStats;
import org.jpmml.converter.Feature;
import org.jpmml.converter.OutlierDecorator;
import org.jpmml.converter.WildcardFeature;
import org.jpmml.python.ClassDictUtil;
import org.jpmml.python.TypeInfo;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn.HasNumberOfFeatures;

public class ContinuousDomain extends Domain {

	public ContinuousDomain(String module, String name){
		super(module, name);
	}

	@Override
	public int getNumberOfFeatures(){
		Boolean withData = getWithData();

		if(withData){
			int[] dataMinShape = getDataMinShape();
			int[] dataMaxShape = getDataMaxShape();

			if(dataMinShape[0] == dataMaxShape[0]){
				return dataMinShape[0];
			}
		}

		return HasNumberOfFeatures.UNKNOWN;
	}

	@Override
	public OpType getOpType(){
		return OpType.CONTINUOUS;
	}

	@Override
	public DataType getDataType(){
		TypeInfo dtype = getDType();

		if(dtype != null){
			return dtype.getDataType();
		}

		return DataType.DOUBLE;
	}

	@Override
	public List<Feature> encodeFeatures(List<Feature> features, SkLearnEncoder encoder){
		features = super.encodeFeatures(features, encoder);

		OutlierTreatmentMethod outlierTreatment = parseOutlierTreatment(getOutlierTreatment());

		Number lowValue;
		Number highValue;

		if(outlierTreatment != null){

			switch(outlierTreatment){
				case AS_EXTREME_VALUES:
				case AS_MISSING_VALUES:
					lowValue = getLowValue();
					highValue = getHighValue();
					break;
				default:
					lowValue = null;
					highValue = null;
			}
		} else

		{
			lowValue = null;
			highValue = null;
		}

		Boolean withData = getWithData();
		Boolean withStatistics = getWithStatistics();

		List<Number> dataMin = null;
		List<Number> dataMax = null;

		if(withData){
			dataMin = getDataMin();
			dataMax = getDataMax();

			ClassDictUtil.checkSize(features, dataMin, dataMax);
		}

		Map<String, ?> counts = null;
		Map<String, ?> numericInfo = null;

		if(withStatistics){
			counts = getCounts();
			numericInfo = getNumericInfo();
		}

		List<Feature> result = new ArrayList<>();

		for(int i = 0; i < features.size(); i++){
			Feature feature = features.get(i);

			WildcardFeature wildcardFeature = asWildcardFeature(feature);

			DataField dataField = wildcardFeature.getField();

			if(outlierTreatment != null){
				encoder.addDecorator(dataField, new OutlierDecorator(outlierTreatment, lowValue, highValue));
			} // End if

			if(withData){
				Interval interval = new Interval(Interval.Closure.CLOSED_CLOSED, dataMin.get(i), dataMax.get(i));

				dataField.addIntervals(interval);

				feature = wildcardFeature.toContinuousFeature();
			} // End if

			if(withStatistics){
				UnivariateStats univariateStats = new UnivariateStats(dataField)
					.setCounts(createCounts(extractMap(counts, i)))
					.setNumericInfo(createNumericInfo(wildcardFeature.getDataType(), extractMap(numericInfo, i)));

				encoder.addUnivariateStats(univariateStats);
			}

			result.add(feature);
		}

		return result;
	}

	@Override
	public int[] getArrayShape(String name){
		int[] shape = super.getArrayShape(name);

		// XXX
		if(shape.length == 0){
			return new int[]{1};
		}

		return shape;
	}

	public String getOutlierTreatment(){
		return getOptionalEnum("outlier_treatment", this::getOptionalString, Arrays.asList(ContinuousDomain.OUTLIERTREATMENT_AS_IS, ContinuousDomain.OUTLIERTREATMENT_AS_EXTREME_VALUES, ContinuousDomain.OUTLIERTREATMENT_AS_MISSING_VALUES));
	}

	public Number getLowValue(){
		return getNumber("low_value");
	}

	public Number getHighValue(){
		return getNumber("high_value");
	}

	public List<Number> getDataMin(){
		return getNumberArray("data_min_");
	}

	public int[] getDataMinShape(){
		return getArrayShape("data_min_", 1);
	}

	public List<Number> getDataMax(){
		return getNumberArray("data_max_");
	}

	public int[] getDataMaxShape(){
		return getArrayShape("data_max_", 1);
	}

	public Map<String, ?> getNumericInfo(){
		return getDict("numeric_info_");
	}

	static
	public OutlierTreatmentMethod parseOutlierTreatment(String outlierTreatment){

		if(outlierTreatment == null){
			return null;
		}

		switch(outlierTreatment){
			case ContinuousDomain.OUTLIERTREATMENT_AS_IS:
				return OutlierTreatmentMethod.AS_IS;
			case ContinuousDomain.OUTLIERTREATMENT_AS_MISSING_VALUES:
				return OutlierTreatmentMethod.AS_MISSING_VALUES;
			case ContinuousDomain.OUTLIERTREATMENT_AS_EXTREME_VALUES:
				return OutlierTreatmentMethod.AS_EXTREME_VALUES;
			default:
				throw new IllegalArgumentException(outlierTreatment);
		}
	}

	static
	public NumericInfo createNumericInfo(DataType dataType, Map<String, ?> values){
		NumericInfo numericInfo = new NumericInfo()
			.setMinimum(selectValue(values, "minimum"))
			.setMaximum(selectValue(values, "maximum"))
			.setMean(selectValue(values, "mean"))
			.setStandardDeviation(selectValue(values, "standardDeviation"))
			.setMedian(selectValue(values, "median"))
			.setInterQuartileRange(selectValue(values, "interQuartileRange"));

		return numericInfo;
	}

	private static final String OUTLIERTREATMENT_AS_IS = "as_is";
	private static final String OUTLIERTREATMENT_AS_EXTREME_VALUES = "as_extreme_values";
	private static final String OUTLIERTREATMENT_AS_MISSING_VALUES = "as_missing_values";
}