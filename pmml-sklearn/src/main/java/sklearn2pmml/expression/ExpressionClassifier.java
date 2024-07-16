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

import java.util.ArrayList;
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
import org.jpmml.converter.ModelEncoder;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.Schema;
import org.jpmml.converter.regression.RegressionModelUtil;
import org.jpmml.python.DataFrameScope;
import org.jpmml.python.Scope;
import sklearn.Classifier;
import sklearn2pmml.util.EvaluatableUtil;

public class ExpressionClassifier extends Classifier {

	public ExpressionClassifier(String module, String name){
		super(module, name);
	}

	@Override
	public RegressionModel encodeModel(Schema schema){
		Map<?, ?> classExprs = getClassExprs();
		RegressionModel.NormalizationMethod normalizationMethod = parseNormalizationMethod(getNormalizationMethod());

		ModelEncoder encoder = schema.getEncoder();
		CategoricalLabel categoricalLabel = (CategoricalLabel)schema.getLabel();
		List<? extends Feature> features = schema.getFeatures();

		Scope scope = new DataFrameScope("X", features, encoder);

		Map<Object, RegressionTable> categoryRegressionTables = new LinkedHashMap<>();

		Collection<? extends Map.Entry<?, ?>> entries = classExprs.entrySet();
		for(Map.Entry<?, ?> entry : entries){
			Object category = entry.getKey();
			Object expr = entry.getValue();

			org.dmg.pmml.Expression pmmlExpression = EvaluatableUtil.translateExpression(expr, scope);

			Feature exprFeature = ExpressionUtil.toFeature(FieldNameUtil.create("expression", category), pmmlExpression, encoder);

			RegressionTable regressionTable = RegressionModelUtil.createRegressionTable(Collections.singletonList(exprFeature), Collections.singletonList(1d), 0d);

			categoryRegressionTables.put(category, regressionTable);
		}

		List<?> categories = categoricalLabel.getValues();

		List<RegressionTable> regressionTables;

		switch(normalizationMethod){
			case NONE:
				if(categoricalLabel.size() == 2){
					regressionTables = encodeBinaryClassifier(categories, categoryRegressionTables);
				} else

				if(categoricalLabel.size() >= 3){
					List<RegressionTable> activeRegressionTables = encodeMultinomialClassifier(categories.subList(0, categories.size() - 1), categoryRegressionTables);

					RegressionTable passiveRegressionTable = RegressionModelUtil.createRegressionTable(Collections.emptyList(), Collections.emptyList(), null)
						.setTargetCategory(categories.get(categories.size() - 1));

					regressionTables = new ArrayList<>(activeRegressionTables);
					regressionTables.add(passiveRegressionTable);
				} else

				{
					throw new IllegalArgumentException();
				}
				break;
			case LOGIT:
				if(categoricalLabel.size() == 2){
					regressionTables = encodeBinaryClassifier(categories, categoryRegressionTables);
				} else

				{
					throw new IllegalArgumentException();
				}
				break;
			case SOFTMAX:
			case SIMPLEMAX:
				if(categoricalLabel.size() >= 2){
					regressionTables = encodeMultinomialClassifier(categories, categoryRegressionTables);
				} else

				{
					throw new IllegalArgumentException();
				}
				break;
			default:
				throw new IllegalArgumentException();
		}

		RegressionModel regressionModel = new RegressionModel(MiningFunction.CLASSIFICATION, ModelUtil.createMiningSchema(categoricalLabel), regressionTables)
			.setNormalizationMethod(normalizationMethod);

		encodePredictProbaOutput(regressionModel, DataType.DOUBLE, categoricalLabel);

		return regressionModel;
	}

	public Map<?, ?> getClassExprs(){
		return getDict("class_exprs");
	}

	public String getNormalizationMethod(){
		return getEnum("normalization_method", this::getString, Arrays.asList(ExpressionClassifier.NORMALIZATIONMETHOD_NONE, ExpressionClassifier.NORMALIZATIONMETHOD_LOGIT, ExpressionClassifier.NORMALIZATIONMETHOD_SIMPLEMAX, ExpressionClassifier.NORMALIZATIONMETHOD_SOFTMAX));
	}

	static
	private RegressionModel.NormalizationMethod parseNormalizationMethod(String normalizationMethod){

		switch(normalizationMethod){
			case ExpressionClassifier.NORMALIZATIONMETHOD_NONE:
				return RegressionModel.NormalizationMethod.NONE;
			case ExpressionClassifier.NORMALIZATIONMETHOD_LOGIT:
				return RegressionModel.NormalizationMethod.LOGIT;
			case ExpressionClassifier.NORMALIZATIONMETHOD_SIMPLEMAX:
				return RegressionModel.NormalizationMethod.SIMPLEMAX;
			case ExpressionClassifier.NORMALIZATIONMETHOD_SOFTMAX:
				return RegressionModel.NormalizationMethod.SOFTMAX;
			default:
				throw new IllegalArgumentException(normalizationMethod);
		}
	}

	static
	private List<RegressionTable> encodeBinaryClassifier(List<?> categories, Map<?, RegressionTable> categoryRegressionTables){

		if(categoryRegressionTables.size() != 1){
			throw new IllegalArgumentException();
		}

		Map.Entry<?, RegressionTable> entry = Iterables.getOnlyElement(categoryRegressionTables.entrySet());

		Object activeCategory = entry.getKey();
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

		RegressionTable activeRegressionTable = entry.getValue()
			.setTargetCategory(activeCategory);

		RegressionTable passiveRegressionTable = RegressionModelUtil.createRegressionTable(Collections.emptyList(), Collections.emptyList(), null)
			.setTargetCategory(passiveCategory);

		return Arrays.asList(activeRegressionTable, passiveRegressionTable);
	}

	static
	private List<RegressionTable> encodeMultinomialClassifier(List<?> categories, Map<?, RegressionTable> categoryRegressionTables){

		if(categoryRegressionTables.size() != categories.size() || !(categoryRegressionTables.keySet()).containsAll(categories)){
			throw new IllegalArgumentException();
		}

		return categories.stream()
			.map(category -> {
				return categoryRegressionTables.get(category)
					.setTargetCategory(category);
			})
			.collect(Collectors.toList());
	}

	private static final String NORMALIZATIONMETHOD_LOGIT = "logit";
	private static final String NORMALIZATIONMETHOD_NONE = "none";
	private static final String NORMALIZATIONMETHOD_SIMPLEMAX = "simplemax";
	private static final String NORMALIZATIONMETHOD_SOFTMAX = "softmax";
}