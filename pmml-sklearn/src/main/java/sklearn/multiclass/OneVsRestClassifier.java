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
package sklearn.multiclass;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.dmg.pmml.DataType;
import org.dmg.pmml.Model;
import org.dmg.pmml.Output;
import org.dmg.pmml.regression.RegressionModel;
import org.jpmml.converter.CategoricalLabel;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.Schema;
import org.jpmml.converter.SchemaUtil;
import org.jpmml.converter.ValueUtil;
import org.jpmml.converter.mining.MiningModelUtil;
import sklearn.Classifier;
import sklearn.EstimatorCheckException;
import sklearn.HasClasses;
import sklearn.HasEstimatorEnsemble;
import sklearn.HasMultiDecisionFunctionField;
import sklearn.HasNumberOfFeatures;
import sklearn.SkLearnClassifier;
import sklearn.StepUtil;

public class OneVsRestClassifier extends SkLearnClassifier implements HasEstimatorEnsemble<Classifier>, HasMultiDecisionFunctionField {

	public OneVsRestClassifier(String module, String name){
		super(module, name);
	}

	@Override
	public int getNumberOfFeatures(){
		int numberOfFeatures = super.getNumberOfFeatures();

		if(numberOfFeatures != HasNumberOfFeatures.UNKNOWN){
			return numberOfFeatures;
		}

		List<Classifier> estimators = getEstimators();

		return StepUtil.getNumberOfFeatures(estimators);
	}

	@Override
	public Model encodeModel(Schema schema){
		List<Classifier> estimators = getEstimators();
		Boolean multilabel = getMultilabel();

		if(multilabel){
			throw new IllegalArgumentException();
		}

		CategoricalLabel categoricalLabel = (CategoricalLabel)schema.getLabel();

		if(estimators.size() == 1){
			SchemaUtil.checkSize(2, categoricalLabel);

			Classifier estimator = estimators.get(0);

			if(!estimator.hasProbabilityDistribution()){
				throw new EstimatorCheckException(estimator, HasClasses.class);
			}

			return estimator.encode(schema);
		} else

		if(estimators.size() >= 2){
			SchemaUtil.checkSize(estimators.size(), categoricalLabel);

			List<Model> models = new ArrayList<>();

			for(int i = 0; i < estimators.size(); i++){
				Classifier estimator = estimators.get(i);

				if(!estimator.hasProbabilityDistribution()){
					throw new EstimatorCheckException(estimator, HasClasses.class);
				}

				Output output = new Output()
					.addOutputFields(ModelUtil.createProbabilityField(getMultiDecisionFunctionField(categoricalLabel.getValue(i)), DataType.DOUBLE, categoricalLabel.getValue(i)));

				CategoricalLabel segmentCategoricalLabel = new CategoricalLabel(DataType.STRING, Arrays.asList("(other)", ValueUtil.asString(categoricalLabel.getValue(i))));

				Schema segmentSchema = schema.toRelabeledSchema(segmentCategoricalLabel);

				Model model = estimator.encode(segmentSchema)
					.setOutput(output);

				models.add(model);
			}

			return MiningModelUtil.createClassification(models, RegressionModel.NormalizationMethod.SIMPLEMAX, true, schema);
		} else

		{
			throw new IllegalArgumentException();
		}
	}

	@Override
	public List<Classifier> getEstimators(){
		return getClassifierList("estimators_");
	}

	public Boolean getMultilabel(){
		return getOptionalBoolean("multilabel_", Boolean.FALSE);
	}
}