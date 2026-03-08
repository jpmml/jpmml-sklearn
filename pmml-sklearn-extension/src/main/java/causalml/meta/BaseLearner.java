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
package causalml.meta;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

import com.google.common.collect.Iterables;
import org.dmg.pmml.DataType;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.Model;
import org.dmg.pmml.Target;
import org.dmg.pmml.Targets;
import org.dmg.pmml.mining.MiningModel;
import org.dmg.pmml.mining.Segmentation;
import org.dmg.pmml.mining.Segmentation.MultipleModelMethod;
import org.jpmml.converter.CategoricalLabel;
import org.jpmml.converter.ContinuousLabel;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.Schema;
import org.jpmml.converter.ValueUtil;
import org.jpmml.converter.mining.MiningModelUtil;
import org.jpmml.python.ClassDictUtil;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn.Classifier;
import sklearn.Estimator;
import sklearn.EstimatorCastFunction;
import sklearn.Regressor;

abstract
public class BaseLearner<E extends Estimator> extends Regressor {

	public BaseLearner(String module, String name){
		super(module, name);
	}

	abstract
	public Class<? extends E> getEstimatorClass();

	abstract
	public Model encodeEstimator(Role role, E estimator, Schema schema);

	protected MiningModel encodeBinaryModel(Model treatmentModel, Model controlModel, Schema schema){
		Targets targets = controlModel.getTargets();

		if(targets != null){
			Target target = Iterables.getOnlyElement(targets);

			Number rescaleFactor = target.getRescaleFactor();
			Number rescaleConstant = target.getRescaleConstant();

			if(rescaleFactor.doubleValue() != 0d){
				target.setRescaleFactor((Number)ValueUtil.toNegative(rescaleFactor));
			} // End if

			if(rescaleConstant.doubleValue() != 0d){
				target.setRescaleConstant((Number)ValueUtil.toNegative(rescaleConstant));
			}
		} else

		{
			ContinuousLabel continuousLabel = new ContinuousLabel(null, DataType.DOUBLE);

			targets = ModelUtil.createRescaleTargets(-1, null, continuousLabel);

			controlModel.setTargets(targets);
		}

		MiningModel miningModel = new MiningModel(MiningFunction.REGRESSION, ModelUtil.createMiningSchema(schema))
			.setSegmentation(MiningModelUtil.createSegmentation(MultipleModelMethod.SUM, Segmentation.MissingPredictionTreatment.RETURN_MISSING, Arrays.asList(treatmentModel, controlModel)));

		return miningModel;
	}

	public String getControlName(){
		return getString("control_name");
	}

	public Map<String, E> getModels(String name){
		Map<String, ?> models = getDict(name);

		Class<? extends E> estimatorClazz = getEstimatorClass();

		Function<Object, E> valueFunction = new EstimatorCastFunction<E>(estimatorClazz){

			@Override
			protected String formatMessage(Object object){
				return "The model object (" + ClassDictUtil.formatClass(object) + ") is not a supported Estimator";
			}
		};

		Map<String, E> result = (models.entrySet()).stream()
			.collect(Collectors.toMap(entry -> entry.getKey(), entry -> valueFunction.apply(entry.getValue())));

		return result;
	}

	public List<String> getTreatmentGroups(){
		return getStringArray("t_groups");
	}

	static
	protected Schema toClassifierSchema(Classifier classifier, Schema schema){
		SkLearnEncoder encoder = (SkLearnEncoder)schema.getEncoder();

		CategoricalLabel categoricalLabel = ((CategoricalLabel)classifier.encodeLabel(Collections.singletonList(null), encoder))
			.expectCardinality(2);

		return schema.toRelabeledSchema(categoricalLabel);
	}

	static
	protected Schema toRegressorSchema(Regressor regressor, Schema schema){
		SkLearnEncoder encoder = (SkLearnEncoder)schema.getEncoder();

		ContinuousLabel continuousLabel = (ContinuousLabel)regressor.encodeLabel(Collections.singletonList(null), encoder);

		return schema.toRelabeledSchema(continuousLabel);
	}
}