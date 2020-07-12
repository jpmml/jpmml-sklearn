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
package sklearn2pmml.decoration;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import org.jpmml.converter.Feature;
import org.jpmml.python.ClassDictUtil;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn.MultiTransformer;

public class MultiDomain extends MultiTransformer {

	public MultiDomain(String module, String name){
		super(module, name);
	}

	@Override
	public List<Feature> encodeFeatures(List<Feature> features, SkLearnEncoder encoder){
		List<? extends Domain> domains = getDomains();

		ClassDictUtil.checkSize(domains, features);

		List<Feature> result = new ArrayList<>();

		for(int i = 0; i < features.size(); i++){
			Domain domain = domains.get(i);
			Feature feature = features.get(i);

			List<Feature> domainFeatures = Collections.singletonList(feature);

			if(domain != null){
				domainFeatures = domain.encode(domainFeatures, encoder);
			}

			result.addAll(domainFeatures);
		}

		return result;
	}

	public List<? extends Domain> getDomains(){
		return getList("domains", Domain.class);
	}
}