/*
 * Copyright (c) 2019 Villu Ruusmann
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

import java.time.LocalDateTime;
import java.time.ZoneId;
import java.util.ArrayList;
import java.util.GregorianCalendar;
import java.util.List;
import java.util.TimeZone;

import org.dmg.pmml.Apply;
import org.dmg.pmml.DataType;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.OpType;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.Feature;
import org.jpmml.converter.FeatureUtil;
import org.jpmml.converter.ObjectFeature;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn.Transformer;

abstract
public class DurationTransformer extends Transformer {

	public DurationTransformer(String module, String name){
		super(module, name);
	}

	abstract
	public String getFunction();

	@Override
	public DataType getDataType(){
		return DataType.INTEGER;
	}

	@Override
	public List<Feature> encodeFeatures(List<Feature> features, SkLearnEncoder encoder){
		GregorianCalendar epoch = getEpoch();
		String function = getFunction();

		ZoneId epochZoneId = ZoneId.systemDefault();

		TimeZone epochTimeZone = epoch.getTimeZone();
		if(epochTimeZone != null){
			epochZoneId = epochTimeZone.toZoneId();
		}

		LocalDateTime epochDateTime = LocalDateTime.ofInstant(epoch.toInstant(), epochZoneId);
		if(epochDateTime.getMonthValue() != 1 || epochDateTime.getDayOfMonth() != 1){
			throw new IllegalArgumentException(String.valueOf(epochDateTime));
		}

		List<Feature> result = new ArrayList<>();

		for(int i = 0; i < features.size(); i++){
			ObjectFeature objectFeature = (ObjectFeature)features.get(i);

			Apply apply = PMMLUtil.createApply(function, objectFeature.ref(), PMMLUtil.createConstant(epochDateTime.getYear(), DataType.INTEGER));

			DerivedField derivedField = encoder.createDerivedField(FeatureUtil.createName("days_since_year", objectFeature), OpType.CONTINUOUS, DataType.INTEGER, apply);

			result.add(new ContinuousFeature(encoder, derivedField));
		}

		return result;
	}

	public GregorianCalendar getEpoch(){
		return get("epoch", GregorianCalendar.class);
	}
}