/*
 * Copyright (c) 2026 Villu Ruusmann
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
package ngboost;

import java.util.Arrays;
import java.util.List;

import org.dmg.pmml.DataType;
import org.dmg.pmml.OpType;
import org.dmg.pmml.mining.MiningModel;
import org.dmg.pmml.regression.RegressionModel;
import org.jpmml.converter.CategoricalLabel;
import org.jpmml.converter.LabelUtil;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.Schema;
import org.jpmml.converter.mining.MiningModelUtil;
import org.jpmml.sklearn.SkLearnException;
import sklearn.Classifier;
import sklearn.Regressor;

public class NGBClassifier extends Classifier {

	public NGBClassifier(String module, String name){
		super(module, name);
	}

	@Override
	public List<?> getClasses(){
		Integer k = getK();

		try {
			return super.getClasses();
		// Getting SkLearnException instead of MissingAttributeException, because this class is a Step subclass
		} catch(SkLearnException se){
			return LabelUtil.createTargetCategories(k);
		}
	}

	@Override
	public MiningModel encodeModel(Schema schema){

		String distName = getDistName();
		switch(distName){
			case NGBClassifier.DIST_CATEGORICAL:
				return encodeCategoricalModel(schema);
			default:
				throw new IllegalArgumentException(distName);
		}
	}

	public MiningModel encodeCategoricalModel(Schema schema){
		List<List<Regressor>> baseModels = getBaseModels();
		List<Number> initParams = getInitParams();
		Number learningRate = getLearningRate();
		List<Number> scalings = getScalings();

		@SuppressWarnings("unused")
		CategoricalLabel categoricalLabel = schema.requireCategoricalLabel()
			.expectCardinality(2);

		Schema segmentSchema = schema.toAnonymousRegressorSchema(DataType.DOUBLE);

		MiningModel locModel = NGBoostUtil.encodeParamModel(0, initParams, baseModels, scalings, learningRate, segmentSchema)
			.setOutput(ModelUtil.createPredictedOutput(NGBoostNames.OUTPUT_LOC, OpType.CONTINUOUS, DataType.DOUBLE));

		return MiningModelUtil.createBinaryLogisticClassification(locModel, 1d, 0d, RegressionModel.NormalizationMethod.LOGIT, true, schema);
	}

	public List<List<Regressor>> getBaseModels(){
		return NGBoostUtil.getBaseModels(this);
	}

	public String getDistName(){
		return getEnum("Dist_name", this::getString, Arrays.asList(NGBClassifier.DIST_CATEGORICAL));
	}

	public List<Number> getInitParams(){
		return getNumberListLike("init_params");
	}

	public Integer getK(){
		return getInteger("K");
	}

	public Number getLearningRate(){
		return getNumber("learning_rate");
	}

	public List<Number> getScalings(){
		return getNumberListLike("scalings");
	}

	private static final String DIST_CATEGORICAL = "Categorical";
}