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

import java.util.Collections;
import java.util.List;
import java.util.Map;

import org.dmg.pmml.OpType;
import org.dmg.pmml.UnivariateStats;
import org.jpmml.converter.Feature;
import org.jpmml.converter.WildcardFeature;
import org.jpmml.sklearn.ClassDictUtil;
import org.jpmml.sklearn.SkLearnEncoder;

public class CategoricalDomain extends DiscreteDomain {

	public CategoricalDomain(String module, String name){
		super(module, name);
	}

	@Override
	public int getNumberOfFeatures(){
		return 1;
	}

	@Override
	public OpType getOpType(){
		return OpType.CATEGORICAL;
	}

	@Override
	public List<Feature> encodeFeatures(List<Feature> features, SkLearnEncoder encoder){
		Boolean withData = getWithData();
		Boolean withStatistics = getWithStatistics();

		ClassDictUtil.checkSize(1, features);

		Feature feature = features.get(0);

		WildcardFeature wildcardFeature = asWildcardFeature(feature);

		if(withData){
			List<?> data = getData();

			feature = wildcardFeature.toCategoricalFeature(data);
		} // End if

		if(withStatistics){
			Map<String, ?> counts = extractMap(getCounts(), 0);
			Object[] discrStats = getDiscrStats();

			UnivariateStats univariateStats = new UnivariateStats()
				.setField(wildcardFeature.getName())
				.setCounts(createCounts(counts))
				.setDiscrStats(createDiscrStats(discrStats));

			encoder.putUnivariateStats(univariateStats);
		}

		return super.encodeFeatures(Collections.singletonList(feature), encoder);
	}
}