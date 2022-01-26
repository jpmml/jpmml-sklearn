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
import java.util.ArrayList;
import java.util.GregorianCalendar;
import java.util.List;

import com.google.common.base.CaseFormat;
import org.dmg.pmml.DataType;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.OpType;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.Feature;
import org.jpmml.converter.ObjectFeature;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.python.CalendarUtil;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn.Transformer;

abstract
public class DurationTransformer extends Transformer {

	public DurationTransformer(String module, String name){
		super(module, name);
	}

	abstract
	public String getPMMLFunction();

	@Override
	public DataType getDataType(){
		return DataType.INTEGER;
	}

	@Override
	public List<Feature> encodeFeatures(List<Feature> features, SkLearnEncoder encoder){
		GregorianCalendar epoch = getEpoch();
		String pmmlFunction = getPMMLFunction();

		LocalDateTime epochDateTime = CalendarUtil.toLocalDateTime(epoch);
		if(epochDateTime.getMonthValue() != 1 || epochDateTime.getDayOfMonth() != 1){
			throw new IllegalArgumentException(String.valueOf(epochDateTime));
		}

		int year = epochDateTime.getYear();

		String function = pmmlFunction;

		if(function.startsWith("date")){
			function = function.substring("date".length());

			function = CaseFormat.UPPER_CAMEL.to(CaseFormat.LOWER_CAMEL, function);
		}

		List<Feature> result = new ArrayList<>();

		for(int i = 0; i < features.size(); i++){
			ObjectFeature objectFeature = (ObjectFeature)features.get(i);

			DerivedField derivedField = encoder.ensureDerivedField(createFieldName(function, objectFeature, year), OpType.CONTINUOUS, DataType.INTEGER, () -> PMMLUtil.createApply(pmmlFunction, objectFeature.ref(), PMMLUtil.createConstant(year, DataType.INTEGER)));

			result.add(new ContinuousFeature(encoder, derivedField));
		}

		return result;
	}

	public GregorianCalendar getEpoch(){
		return get("epoch", GregorianCalendar.class);
	}
}