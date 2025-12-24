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
package sklearn.naive_bayes;

import java.util.ArrayList;
import java.util.List;

import org.dmg.pmml.naive_bayes.BayesInput;
import org.dmg.pmml.naive_bayes.BayesInputs;
import org.dmg.pmml.naive_bayes.PairCounts;
import org.jpmml.converter.CMatrixUtil;
import org.jpmml.converter.CategoricalFeature;
import org.jpmml.converter.Feature;
import org.jpmml.converter.SchemaUtil;
import org.jpmml.python.HasArray;

public class CategoricalNB extends DiscreteNB {

	public CategoricalNB(String module, String name){
		super(module, name);
	}

	@Override
	public BayesInputs encodeBayesInputs(List<?> values, List<? extends Feature> features){
		Number alpha = getAlpha();
		List<HasArray> categoryCount = getCategoryCount();
		List<Integer> numberOfCategories = getNumberOfCategories();

		BayesInputs bayesInputs = new BayesInputs();

		for(int i = 0; i < features.size(); i++){
			Feature feature = features.get(i);
			int numberOfFeatureCategories = numberOfCategories.get(i);

			BayesInput bayesInput;

			if(feature instanceof CategoricalFeature){
				CategoricalFeature categoricalFeature = (CategoricalFeature)feature;

				SchemaUtil.checkCardinality(numberOfFeatureCategories, categoricalFeature);

				HasArray featureCategoryCount = categoryCount.get(i);

				List<PairCounts> pairCounts = new ArrayList<>();

				for(int j = 0; j < numberOfFeatureCategories; j++){
					List<Number> featureValueCounts = (List<Number>)CMatrixUtil.getColumn(featureCategoryCount.getArrayContent(), values.size(), numberOfFeatureCategories, j);

					pairCounts.add(DiscreteNBUtil.encodePairCounts(categoricalFeature.getValue(j), values, alpha, featureValueCounts));
				}

				bayesInput = new BayesInput(categoricalFeature.getName(), null, pairCounts);
			} else

			{
				throw new IllegalArgumentException("Expected a categorical feature, got " + feature);
			}

			bayesInputs.addBayesInputs(bayesInput);
		}

		return bayesInputs;
	}

	public List<HasArray> getCategoryCount(){
		return getArrayList("category_count_");
	}

	public List<Integer> getNumberOfCategories(){
		return getIntegerArray("n_categories_");
	}
}