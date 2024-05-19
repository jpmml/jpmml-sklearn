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

import com.google.common.collect.Iterables;
import org.dmg.pmml.DataType;
import org.dmg.pmml.Model;
import org.dmg.pmml.regression.RegressionModel;
import org.jpmml.converter.CategoricalLabel;
import org.jpmml.converter.Feature;
import org.jpmml.converter.Schema;
import org.jpmml.converter.regression.RegressionModelUtil;
import org.jpmml.python.HasArray;
import sklearn.Classifier;

public class ExplainableBoostingClassifier extends Classifier implements HasExplainableBooster {

	public ExplainableBoostingClassifier(String module, String name){
		super(module, name);
	}

	@Override
	public Model encodeModel(Schema schema){
		List<Number> intercept = getIntercept();
		RegressionModel.NormalizationMethod normalizationMethod = parseLink(getLink());

		CategoricalLabel categoricalLabel = (CategoricalLabel)schema.getLabel();

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

		RegressionModel regressionModel;

		if(categoricalLabel.size() == 2){
			regressionModel = RegressionModelUtil.createBinaryLogisticClassification(features, coefficients, Iterables.getOnlyElement(intercept), normalizationMethod, false, schema);
		} else

		{
			throw new IllegalArgumentException();
		}

		encodePredictProbaOutput(regressionModel, DataType.DOUBLE, categoricalLabel);

		return regressionModel;
	}

	@Override
	public List<List<?>> getBins(){
		return (List)getList("bins_", List.class);
	}

	public List<Number> getIntercept(){
		return getNumberArray("intercept_");
	}

	public String getLink(){
		return getEnum("link_", this::getString, Arrays.asList(ExplainableBoostingClassifier.LINK_LOGIT));
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
			case ExplainableBoostingClassifier.LINK_LOGIT:
				return RegressionModel.NormalizationMethod.LOGIT;
			default:
				throw new IllegalArgumentException(link);
		}
	}

	private static final String LINK_LOGIT = "logit";
}