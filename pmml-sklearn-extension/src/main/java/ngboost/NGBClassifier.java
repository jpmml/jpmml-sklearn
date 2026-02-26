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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import org.dmg.pmml.DataType;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.Model;
import org.dmg.pmml.OpType;
import org.dmg.pmml.PMML;
import org.dmg.pmml.mining.MiningModel;
import org.dmg.pmml.mining.Segmentation.MissingPredictionTreatment;
import org.dmg.pmml.regression.RegressionModel;
import org.dmg.pmml.regression.RegressionTable;
import org.jpmml.converter.CategoricalLabel;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.FieldNameUtil;
import org.jpmml.converter.LabelUtil;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.PMMLEncoder;
import org.jpmml.converter.Schema;
import org.jpmml.converter.mining.MiningModelUtil;
import org.jpmml.converter.regression.RegressionModelUtil;
import org.jpmml.sklearn.Encodable;
import org.jpmml.sklearn.HasSkLearnOptions;
import org.jpmml.sklearn.SkLearnException;
import sklearn.Classifier;
import sklearn.EstimatorUtil;
import sklearn.HasFeatureNamesIn;
import sklearn.Regressor;
import sklearn.tree.HasTreeOptions;
import sklearn.tree.TreeRegressor;
import sklearn.tree.TreeUtil;

public class NGBClassifier extends Classifier implements HasFeatureNamesIn, HasSkLearnOptions, HasTreeOptions, Encodable {

	public NGBClassifier(String module, String name){
		super(module, name);
	}

	@Override
	public int getNumberOfFeatures(){
		return getInteger("n_features");
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

	@Override
	public Schema configureSchema(Schema schema){
		Regressor base = getBase();

		if(base instanceof TreeRegressor){
			return TreeUtil.configureSchema(this, schema);
		}

		return super.configureSchema(schema);
	}

	@Override
	public Model configureModel(Model model){
		Regressor base = getBase();

		if(base instanceof TreeRegressor){
			return TreeUtil.configureModel(this, model);
		}

		return super.configureModel(model);
	}

	@Override
	public PMML encodePMML(){
		return EstimatorUtil.encodePMML(this);
	}

	/**
	 * @see NGBoostNames#OUTPUT_LOC
	 */
	public MiningModel encodeCategoricalModel(Schema schema){
		List<List<Regressor>> baseModels = getBaseModels();
		List<Number> initParams = getInitParams();
		Integer k = getK();
		Number learningRate = getLearningRate();
		List<Number> scalings = getScalings();

		PMMLEncoder encoder = schema.getEncoder();
		CategoricalLabel categoricalLabel = schema.requireCategoricalLabel()
			.expectCardinality(k);

		Schema segmentSchema = schema.toAnonymousRegressorSchema(DataType.DOUBLE);

		if(categoricalLabel.size() == 2){
			MiningModel locModel = NGBoostUtil.encodeParamModel(0, initParams, baseModels, scalings, learningRate, segmentSchema)
				.setOutput(ModelUtil.createPredictedOutput(NGBoostNames.OUTPUT_LOC, OpType.CONTINUOUS, DataType.DOUBLE));

			return MiningModelUtil.createBinaryLogisticClassification(locModel, 1d, 0d, RegressionModel.NormalizationMethod.LOGIT, true, schema);
		} else

		if(categoricalLabel.size() >= 3){
			categoricalLabel.expectCardinality(initParams.size() + 1);

			List<Model> models = new ArrayList<>();

			for(int i = 0; i < (k - 1); i++){
				MiningModel locModel = NGBoostUtil.encodeParamModel(i, initParams, baseModels, scalings, learningRate, segmentSchema)
					.setOutput(ModelUtil.createPredictedOutput(FieldNameUtil.create(NGBoostNames.OUTPUT_LOC, i), OpType.CONTINUOUS, DataType.DOUBLE));

				models.add(locModel);
			}

			RegressionModel regressionModel = new RegressionModel(MiningFunction.CLASSIFICATION, ModelUtil.createMiningSchema(schema), null)
				.setNormalizationMethod(RegressionModel.NormalizationMethod.SOFTMAX)
				.setOutput(ModelUtil.createProbabilityOutput(DataType.DOUBLE, categoricalLabel));

			{
				RegressionTable referenceRegressionTable = new RegressionTable(0d)
					.setTargetCategory(categoricalLabel.getValue(0));

				regressionModel.addRegressionTables(referenceRegressionTable);
			}

			for(int i = 0; i < (k - 1); i++){
				ContinuousFeature locFeature = new ContinuousFeature(encoder, FieldNameUtil.create(NGBoostNames.OUTPUT_LOC, i), DataType.DOUBLE);

				RegressionTable regressionTable = RegressionModelUtil.createRegressionTable(Collections.singletonList(locFeature), Collections.singletonList(1d), 0d)
					.setTargetCategory(categoricalLabel.getValue(i + 1));

				regressionModel.addRegressionTables(regressionTable);
			}

			models.add(regressionModel);

			return MiningModelUtil.createModelChain(models, MissingPredictionTreatment.RETURN_MISSING);
		} else

		{
			throw new IllegalArgumentException();
		}
	}

	public Regressor getBase(){
		return getRegressor("Base");
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