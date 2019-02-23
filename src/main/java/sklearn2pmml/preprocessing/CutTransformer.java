/*
 * Copyright (c) 2018 Villu Ruusmann
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
package sklearn2pmml.preprocessing;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import org.dmg.pmml.DataType;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.Discretize;
import org.dmg.pmml.DiscretizeBin;
import org.dmg.pmml.Interval;
import org.dmg.pmml.OpType;
import org.jpmml.converter.CategoricalFeature;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.Feature;
import org.jpmml.converter.FeatureUtil;
import org.jpmml.converter.ValueUtil;
import org.jpmml.sklearn.ClassDictUtil;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn.Transformer;
import sklearn.TypeUtil;

public class CutTransformer extends Transformer {

	public CutTransformer(String module, String name){
		super(module, name);
	}

	@Override
	public List<Feature> encodeFeatures(List<Feature> features, SkLearnEncoder encoder){
		List<? extends Number> bins = getBins();
		List<?> labels = getLabels();
		Boolean right = getRight();
		Boolean includeLowest = getIncludeLowest();

		ClassDictUtil.checkSize(1, features);

		DataType dataType;

		if(labels != null){
			ClassDictUtil.checkSize(bins.size() - 1, labels);

			dataType = TypeUtil.getDataType(labels, DataType.STRING);
		} else

		{
			dataType = DataType.INTEGER;
		}

		Feature feature = features.get(0);

		Interval.Closure closure = (right ? Interval.Closure.OPEN_CLOSED : Interval.Closure.CLOSED_OPEN);

		ContinuousFeature continuousFeature = feature.toContinuousFeature();

		List<String> categories = new ArrayList<>();

		Discretize discretize = new Discretize(continuousFeature.getName());

		for(int i = 0; i < bins.size() - 1; i++){
			Number leftMargin = bins.get(i);
			Number rightMargin = bins.get(i + 1);

			Interval interval = new Interval(closure)
				.setLeftMargin(formatMargin(leftMargin))
				.setRightMargin(formatMargin(rightMargin));

			if(i == 0 && includeLowest && (interval.getClosure()).equals(Interval.Closure.OPEN_CLOSED)){
				interval.setClosure(Interval.Closure.CLOSED_CLOSED);
			}

			String label;

			if(labels != null){
				label = ValueUtil.formatValue(labels.get(i));
			} else

			{
				label = String.valueOf(i);
			}

			categories.add(label);

			DiscretizeBin discretizeBin = new DiscretizeBin(label, interval);

			discretize.addDiscretizeBins(discretizeBin);
		}

		DerivedField derivedField = encoder.createDerivedField(FeatureUtil.createName("cut", feature), OpType.CATEGORICAL, dataType, discretize);

		return Collections.singletonList(new CategoricalFeature(encoder, derivedField, categories));
	}

	public List<? extends Number> getBins(){
		return getList("bins", Number.class);
	}

	public List<?> getLabels(){
		Object labels = getOptionalObject("labels");

		if(labels == null || (Boolean.FALSE).equals(labels)){
			return null;
		}

		return getList("labels");
	}

	public Boolean getRight(){
		return getBoolean("right");
	}

	public Boolean getIncludeLowest(){
		return getBoolean("include_lowest");
	}

	static
	private Double formatMargin(Number number){
		double value = number.doubleValue();

		if(Double.isInfinite(value)){
			return null;
		}

		return value;
	}
}