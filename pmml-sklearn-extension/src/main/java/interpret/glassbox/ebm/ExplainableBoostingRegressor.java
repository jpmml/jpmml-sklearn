/*
 * Copyright (c) 2024 Villu Ruusmann
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
package interpret.glassbox.ebm;

import java.util.AbstractList;
import java.util.Arrays;
import java.util.List;

import org.dmg.pmml.Model;
import org.dmg.pmml.regression.RegressionModel;
import org.jpmml.converter.Feature;
import org.jpmml.converter.Schema;
import org.jpmml.converter.regression.RegressionModelUtil;
import org.jpmml.python.HasArray;
import sklearn.Regressor;

public class ExplainableBoostingRegressor extends Regressor implements HasExplainableBooster {

	public ExplainableBoostingRegressor(String module, String name){
		super(module, name);
	}

	@Override
	public Model encodeModel(Schema schema){
		Number intercept = getIntercept();
		RegressionModel.NormalizationMethod normalizationMethod = parseLink(getLink());

		List<Feature> features = ExplainableBoostingUtil.encodeExplainableBooster(this, schema);

		List<Number> coefficients = new AbstractList<Number>(){

			@Override
			public int size(){
				return features.size();
			}

			@Override
			public Number get(int index){
				return 1d;
			}
		};

		return RegressionModelUtil.createRegression(features, coefficients, intercept, normalizationMethod, schema);
	}

	@Override
	public List<List<?>> getBins(){
		return (List)getList("bins_", List.class);
	}

	@Override
	public List<String> getFeatureTypesIn(){
		return getEnumList("feature_types_in_", this::getStringList, Arrays.asList(ExplainableBoostingRegressor.FEATURETYPE_CONTINUOUS, ExplainableBoostingRegressor.FEATURETYPE_NOMINAL));
	}

	public Number getIntercept(){
		return getNumber("intercept_");
	}

	public String getLink(){
		return getEnum("link_", this::getString, Arrays.asList(ExplainableBoostingRegressor.LINK_IDENTITY));
	}

	@Override
	public List<Object[]> getTermFeatures(){
		return getTupleList("term_features_");
	}

	@Override
	public List<HasArray> getTermScores(){
		return getArrayList("term_scores_");
	}

	static
	private RegressionModel.NormalizationMethod parseLink(String link){

		switch(link){
			case ExplainableBoostingRegressor.LINK_IDENTITY:
				return RegressionModel.NormalizationMethod.NONE;
			default:
				throw new IllegalArgumentException(link);
		}
	}

	private static final String LINK_IDENTITY = "identity";
}