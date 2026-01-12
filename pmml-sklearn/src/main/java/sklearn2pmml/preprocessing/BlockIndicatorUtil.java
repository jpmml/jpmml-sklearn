/*
 * Copyright (c) 2025 Villu Ruusmann
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

import java.util.List;

import com.google.common.base.Function;
import com.google.common.collect.Lists;
import org.dmg.pmml.BlockIndicator;
import org.jpmml.converter.ExceptionUtil;
import org.jpmml.converter.Feature;
import org.jpmml.converter.FeatureUtil;
import org.jpmml.python.ClassDictUtil;
import org.jpmml.sklearn.SkLearnException;

public class BlockIndicatorUtil {

	private BlockIndicatorUtil(){
	}

	static
	public List<Feature> selectFeatures(List<?> blockIndiators, List<Feature> features){
		Function<Object, Feature> castFunction = new Function<Object, Feature>(){

			@Override
			public Feature apply(Object object){

				if(object instanceof String){
					String column = (String)object;

					Feature feature = FeatureUtil.findFeature(features, column);
					if(feature != null){
						return feature;
					}

					throw new SkLearnException("Column " + ExceptionUtil.formatName(column) + " not in " + ExceptionUtil.formatNameList(features));
				} else

				if(object instanceof Integer){
					Integer index = (Integer)object;

					return features.get(index);
				} else

				{
					throw new SkLearnException("The block indicator object (" + ClassDictUtil.formatClass(object) + ") is not a string nor integer");
				}
			}
		};

		return Lists.transform(blockIndiators, castFunction);
	}

	static
	public BlockIndicator[] toBlockIndicators(List<Feature> features){
		return features.stream()
			.map(feature -> new BlockIndicator(feature.getName()))
			.toArray(size -> new BlockIndicator[size]);
	}
}