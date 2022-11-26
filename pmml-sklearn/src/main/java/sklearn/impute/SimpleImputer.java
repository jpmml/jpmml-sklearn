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
package sklearn.impute;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import org.dmg.pmml.DataType;
import org.dmg.pmml.MissingValueTreatmentMethod;
import org.dmg.pmml.OpType;
import org.jpmml.converter.Feature;
import org.jpmml.converter.TypeUtil;
import org.jpmml.converter.ValueUtil;
import org.jpmml.python.ClassDictUtil;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn.Transformer;

public class SimpleImputer extends Transformer {

	public SimpleImputer(String module, String name){
		super(module, name);
	}

	@Override
	public OpType getOpType(){
		String strategy = getStrategy();

		switch(strategy){
			case "constant":
				DataType dataType = getDataType();

				return TypeUtil.getOpType(dataType);
			case "mean":
			case "median":
				return OpType.CONTINUOUS;
			case "most_frequent":
				return OpType.CATEGORICAL;
			default:
				throw new IllegalArgumentException(strategy);
		}
	}

	@Override
	public DataType getDataType(){
		String strategy = getStrategy();
		List<?> statistics = getStatistics();

		switch(strategy){
			case "constant":
				return TypeUtil.getDataType(statistics, DataType.STRING);
			case "mean":
			case "median":
				return DataType.DOUBLE;
			case "most_frequent":
				return TypeUtil.getDataType(statistics, DataType.STRING);
			default:
				throw new IllegalArgumentException(strategy);
		}
	}

	@Override
	public int getNumberOfFeatures(){
		int[] shape = getStatisticsShape();

		return shape[0];
	}

	@Override
	public List<Feature> encodeFeatures(List<Feature> features, SkLearnEncoder encoder){
		Boolean addIndicator = getAddIndicator();
		Object missingValues = getMissingValues();
		List<?> statistics = getStatistics();
		String strategy = getStrategy();

		ClassDictUtil.checkSize(features, statistics);

		if(ValueUtil.isNaN(missingValues)){
			missingValues = null;
		}

		MissingValueTreatmentMethod missingValueTreatment = parseStrategy(strategy);

		List<Feature> indicatorFeatures = new ArrayList<>();

		List<Feature> result = new ArrayList<>();

		for(int i = 0; i < features.size(); i++){
			Feature feature = features.get(i);
			Object statistic = statistics.get(i);

			if(addIndicator){
				Feature indicatorFeature = ImputerUtil.encodeIndicatorFeature(this, feature, missingValues, encoder);

				indicatorFeatures.add(indicatorFeature);
			}

			feature = ImputerUtil.encodeFeature(this, feature, addIndicator, missingValues, statistic, missingValueTreatment, encoder);

			result.add(feature);
		}

		if(addIndicator){
			result.addAll(indicatorFeatures);
		}

		return result;
	}

	public Boolean getAddIndicator(){
		return getOptionalBoolean("add_indicator", Boolean.FALSE);
	}

	public Object getMissingValues(){
		return getOptionalScalar("missing_values");
	}

	public List<?> getStatistics(){

		if(!containsKey("statistics_")){
			return Collections.emptyList();
		}

		return getArray("statistics_");
	}

	public int[] getStatisticsShape(){

		if(!containsKey("statistics_")){
			return new int[]{0};
		}

		return getArrayShape("statistics_", 1);
	}

	public String getStrategy(){
		return getString("strategy");
	}

	static
	private MissingValueTreatmentMethod parseStrategy(String strategy){

		switch(strategy){
			case "constant":
				return MissingValueTreatmentMethod.AS_VALUE;
			case "mean":
				return MissingValueTreatmentMethod.AS_MEAN;
			case "median":
				return MissingValueTreatmentMethod.AS_MEDIAN;
			case "most_frequent":
				return MissingValueTreatmentMethod.AS_MODE;
			default:
				throw new IllegalArgumentException(strategy);
		}
	}
}
