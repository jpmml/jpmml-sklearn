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
import org.jpmml.converter.ExceptionUtil;
import org.jpmml.converter.ExpressionUtil;
import org.jpmml.converter.Feature;
import org.jpmml.converter.ObjectFeature;
import org.jpmml.converter.UnsupportedFeatureException;
import org.jpmml.python.Attribute;
import org.jpmml.python.CalendarUtil;
import org.jpmml.python.InvalidAttributeException;
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
		if(epochDateTime.getDayOfMonth() != 1 || epochDateTime.getMonthValue() != 1){
			throw new InvalidAttributeException("Date attribute " + ExceptionUtil.formatName("epoch") + " must be set to the 1st of January of some year", new Attribute(this, "epoch"));
		}

		int year = epochDateTime.getYear();

		String function = pmmlFunction;

		if(function.startsWith("date")){
			function = function.substring("date".length());

			function = CaseFormat.UPPER_CAMEL.to(CaseFormat.LOWER_CAMEL, function);
		}

		List<Feature> result = new ArrayList<>();

		for(int i = 0; i < features.size(); i++){
			Feature feature = features.get(i);

			if(!(feature instanceof ObjectFeature)){
				throw new UnsupportedFeatureException("Expected a date-type object feature, got " + feature.typeString());
			}

			ObjectFeature objectFeature = (ObjectFeature)feature;

			DerivedField derivedField = encoder.ensureDerivedField(createFieldName(function, objectFeature, year), OpType.CONTINUOUS, DataType.INTEGER, () -> ExpressionUtil.createApply(pmmlFunction, objectFeature.ref(), ExpressionUtil.createConstant(DataType.INTEGER, year)));

			result.add(new ContinuousFeature(encoder, derivedField));
		}

		return result;
	}

	public GregorianCalendar getEpoch(){
		return get("epoch", GregorianCalendar.class);
	}
}