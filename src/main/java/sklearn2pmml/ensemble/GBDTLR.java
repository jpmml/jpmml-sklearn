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
package sklearn2pmml.ensemble;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import org.dmg.pmml.DataType;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.Model;
import org.dmg.pmml.OpType;
import org.dmg.pmml.ScoreDistribution;
import org.dmg.pmml.Visitor;
import org.dmg.pmml.VisitorAction;
import org.dmg.pmml.mining.MiningModel;
import org.dmg.pmml.mining.Segmentation;
import org.dmg.pmml.regression.RegressionModel;
import org.dmg.pmml.tree.Node;
import org.dmg.pmml.tree.TreeModel;
import org.jpmml.converter.CategoricalLabel;
import org.jpmml.converter.ContinuousLabel;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.Schema;
import org.jpmml.converter.SchemaUtil;
import org.jpmml.converter.ValueUtil;
import org.jpmml.converter.mining.MiningModelUtil;
import org.jpmml.model.visitors.AbstractVisitor;
import org.jpmml.sklearn.ClassDictUtil;
import sklearn.Classifier;
import sklearn.linear_model.logistic.LogisticRegression;
import sklearn.preprocessing.MultiOneHotEncoder;
import sklearn.tree.HasTreeOptions;

public class GBDTLR extends Classifier {

	public GBDTLR(String module, String name){
		super(module, name);
	}

	@Override
	public List<?> getClasses(){
		Classifier gbdt = getGBDT();

		return gbdt.getClasses();
	}

	@Override
	public Model encodeModel(Schema schema){
		Classifier gbdt = getGBDT();
		LogisticRegression lr = getLR();
		MultiOneHotEncoder ohe = getOHE();

		CategoricalLabel categoricalLabel = (CategoricalLabel)schema.getLabel();

		SchemaUtil.checkSize(2, categoricalLabel);

		Map<String, Object> pmmlOptions = new LinkedHashMap<>();
		pmmlOptions.put(HasTreeOptions.OPTION_COMPACT, Boolean.FALSE);
		pmmlOptions.put(HasTreeOptions.OPTION_FLAT, Boolean.FALSE);
		pmmlOptions.put(HasTreeOptions.OPTION_NODE_ID, Boolean.TRUE);

		gbdt.setPMMLOptions(pmmlOptions);

		Model model = gbdt.encodeModel(schema);

		List<TreeModel> treeModels = new ArrayList<>();

		Visitor modelVisitor = new AbstractVisitor(){

			@Override
			public VisitorAction visit(TreeModel treeModel){
				treeModels.add(treeModel);

				return super.visit(treeModel);
			}
		};
		modelVisitor.applyTo(model);

		List<? extends Number> coef = lr.getCoef();
		List<? extends Number> intercept = lr.getIntercept();

		List<List<?>> treeCategories = ohe.getCategories();

		ClassDictUtil.checkSize(treeModels, treeCategories);

		List<Map<Integer, Number>> treeNodeScores = new ArrayList<>();

		int coefOffset = 0;

		for(List<?> treeCategory : treeCategories){
			Map<Integer, Number> nodeScores = new LinkedHashMap<>();

			for(int i = 0; i < treeCategory.size(); i++){
				Integer id = ValueUtil.asInteger((Number)treeCategory.get(i));
				Number score = coef.get(coefOffset + i);

				nodeScores.put(id, score);
			}

			treeNodeScores.add(nodeScores);

			coefOffset += treeCategory.size();
		}

		ClassDictUtil.checkSize(coefOffset, coef);
		ClassDictUtil.checkSize(1, intercept);

		for(int i = 0; i < treeModels.size(); i++){
			TreeModel treeModel = treeModels.get(i);
			Map<Integer, Number> nodeScores = treeNodeScores.get(i);

			treeModel
				.setMiningFunction(MiningFunction.REGRESSION)
				.setMathContext(null);

			Visitor treeModelVisitor = new AbstractVisitor(){

				@Override
				public VisitorAction visit(Node node){
					Object id = node.getId();

					if(id instanceof String){
						String string = (String)id;

						id = Integer.parseInt(string);
					} else

					if(id instanceof Number){
						Number number = (Number)id;

						id = ValueUtil.asInteger(number);
					} else

					{
						throw new IllegalArgumentException(String.valueOf(id));
					} // End if

					if(node.hasScoreDistributions()){
						List<ScoreDistribution> scoreDistributions = node.getScoreDistributions();

						scoreDistributions.clear();
					}

					Number score = nodeScores.get((Integer)id);

					node
						//.setId(null)
						.setScore(score);

					return super.visit(node);
				}
			};
			treeModelVisitor.applyTo(treeModel);
		}

		Schema segmentSchema = (schema.toAnonymousRegressorSchema(DataType.DOUBLE)).toEmptySchema();

		ContinuousLabel continuousLabel = (ContinuousLabel)segmentSchema.getLabel();

		MiningModel miningModel = new MiningModel(MiningFunction.REGRESSION, ModelUtil.createMiningSchema(continuousLabel))
			.setSegmentation(MiningModelUtil.createSegmentation(Segmentation.MultipleModelMethod.SUM, treeModels))
			.setTargets(ModelUtil.createRescaleTargets(null, intercept.get(0), continuousLabel))
			.setOutput(ModelUtil.createPredictedOutput(FieldName.create("decisionFunction"), OpType.CONTINUOUS, DataType.DOUBLE));

		return MiningModelUtil.createBinaryLogisticClassification(miningModel, 1d, 0d, RegressionModel.NormalizationMethod.LOGIT, true, schema);
	}

	public Classifier getGBDT(){
		return get("gbdt_", Classifier.class);
	}

	public LogisticRegression getLR(){
		return get("lr_", LogisticRegression.class);
	}

	public MultiOneHotEncoder getOHE(){
		return get("ohe_", MultiOneHotEncoder.class);
	}
}