/*
 * Copyright (c) 2023 Villu Ruusmann
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
package sklearn2pmml.expression;

import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import com.google.common.collect.Iterables;
import org.dmg.pmml.DataType;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.regression.RegressionModel;
import org.dmg.pmml.regression.RegressionTable;
import org.jpmml.converter.CategoricalLabel;
import org.jpmml.converter.Feature;
import org.jpmml.converter.FieldNameUtil;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.PMMLEncoder;
import org.jpmml.converter.Schema;
import org.jpmml.converter.regression.RegressionModelUtil;
import org.jpmml.python.DataFrameScope;
import org.jpmml.python.Scope;
import sklearn.Classifier;
import sklearn2pmml.util.EvaluatableUtil;
import sklearn2pmml.util.Expression;

public class ExpressionClassifier extends Classifier {

	public ExpressionClassifier(String module, String name){
		super(module, name);
	}

	@Override
	public RegressionModel encodeModel(Schema schema){
		Map<?, Expression> classExprs = getClassExprs();
		RegressionModel.NormalizationMethod normalizationMethod = parseNormalizationMethod(getNormalizationMethod());

		PMMLEncoder encoder = schema.getEncoder();

		CategoricalLabel categoricalLabel = (CategoricalLabel)schema.getLabel();
		List<? extends Feature> features = schema.getFeatures();

		Scope scope = new DataFrameScope("X", features, encoder);

		Map<Object, RegressionTable> categoryRegressionTables = new LinkedHashMap<>();

		Collection<? extends Map.Entry<?, Expression>> entries = classExprs.entrySet();
		for(Map.Entry<?, Expression> entry : entries){
			Object category = entry.getKey();
			Expression expr = entry.getValue();

			org.dmg.pmml.Expression pmmlExpression = EvaluatableUtil.translateExpression(expr, scope);

			Feature exprFeature = ExpressionUtil.toFeature(FieldNameUtil.create("expression", category), pmmlExpression, encoder);

			RegressionTable regressionTable = RegressionModelUtil.createRegressionTable(Collections.singletonList(exprFeature), Collections.singletonList(1d), 0d);

			categoryRegressionTables.put(category, regressionTable);
		}

		List<?> categories = categoricalLabel.getValues();

		List<RegressionTable> regressionTables;

		switch(normalizationMethod){
			case LOGIT:
				{
					if((categoryRegressionTables.size() != 1) || (categories.size() != 2)){
						throw new IllegalArgumentException();
					}

					Object activeCategory = Iterables.getOnlyElement(categoryRegressionTables.keySet());
					Object passiveCategory;

					int index = categories.indexOf(activeCategory);
					if(index == 0){
						passiveCategory = categories.get(1);
					} else

					if(index == 1){
						passiveCategory = categories.get(0);
					} else

					{
						throw new IllegalArgumentException();
					}

					RegressionTable activeRegressionTable = categoryRegressionTables.get(activeCategory)
						.setTargetCategory(activeCategory);

					RegressionTable passiveRegressionTable = RegressionModelUtil.createRegressionTable(Collections.emptyList(), Collections.emptyList(), null)
						.setTargetCategory(passiveCategory);

					regressionTables = Arrays.asList(activeRegressionTable, passiveRegressionTable);
				}
				break;
			case SIMPLEMAX:
			case SOFTMAX:
				{
					if((categoryRegressionTables.size() != categories.size()) || !(categoryRegressionTables.keySet()).containsAll(categories)){
						throw new IllegalArgumentException();
					}

					regressionTables = categories.stream()
						.map(category -> {
							return categoryRegressionTables.get(category)
								.setTargetCategory(category);
						})
						.collect(Collectors.toList());
				}
				break;
			default:
				throw new IllegalArgumentException();
		}

		RegressionModel regressionModel = new RegressionModel(MiningFunction.CLASSIFICATION, ModelUtil.createMiningSchema(categoricalLabel), regressionTables)
			.setNormalizationMethod(normalizationMethod)
			.setOutput(ModelUtil.createProbabilityOutput(DataType.DOUBLE, categoricalLabel));

		return regressionModel;
	}

	public Map<?, Expression> getClassExprs(){
		return (Map)getDict("class_exprs");
	}

	public String getNormalizationMethod(){
		return getString("normalization_method");
	}

	static
	private RegressionModel.NormalizationMethod parseNormalizationMethod(String normalizationMethod){

		switch(normalizationMethod){
			case "logit":
				return RegressionModel.NormalizationMethod.LOGIT;
			case "simplemax":
				return RegressionModel.NormalizationMethod.SIMPLEMAX;
			case "softmax":
				return RegressionModel.NormalizationMethod.SOFTMAX;
			default:
				throw new IllegalArgumentException(normalizationMethod);
		}
	}
}