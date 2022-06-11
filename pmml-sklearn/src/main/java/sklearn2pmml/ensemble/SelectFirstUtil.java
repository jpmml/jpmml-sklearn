/*
 * Copyright (c) 2020 Villu Ruusmann
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

import java.util.Collections;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.Model;
import org.dmg.pmml.Output;
import org.dmg.pmml.OutputField;
import org.dmg.pmml.Predicate;
import org.dmg.pmml.ResultFeature;
import org.dmg.pmml.mining.MiningModel;
import org.dmg.pmml.mining.Segment;
import org.dmg.pmml.mining.Segmentation;
import org.jpmml.converter.Feature;
import org.jpmml.converter.Label;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.Schema;
import org.jpmml.converter.mining.MiningModelUtil;
import org.jpmml.model.ReflectionUtil;
import org.jpmml.python.DataFrameScope;
import org.jpmml.python.PredicateTranslator;
import org.jpmml.python.Scope;
import org.jpmml.python.TupleUtil;
import sklearn.Estimator;

public class SelectFirstUtil {

	private SelectFirstUtil(){
	}

	static
	public MiningModel encodeRegressor(List<Object[]> steps, Schema schema){
		return encodeModel(MiningFunction.REGRESSION, steps, schema);
	}

	static
	public MiningModel encodeClassifier(List<Object[]> steps, Schema schema){
		return encodeModel(MiningFunction.CLASSIFICATION, steps, schema);
	}

	static
	private MiningModel encodeModel(MiningFunction miningFunction, List<Object[]> steps, Schema schema){

		if(steps.isEmpty()){
			throw new IllegalArgumentException();
		}

		Label label = schema.getLabel();
		List<? extends Feature> features = schema.getFeatures();

		Segmentation segmentation = new Segmentation(Segmentation.MultipleModelMethod.SELECT_FIRST, null);

		Scope scope = new DataFrameScope("X", features);

		for(Object[] step : steps){
			String name = TupleUtil.extractElement(step, 0, String.class);
			Estimator estimator = TupleUtil.extractElement(step, 1, Estimator.class);
			String predicate = TupleUtil.extractElement(step, 2, String.class);

			if(estimator.getMiningFunction() != miningFunction){
				throw new IllegalArgumentException();
			}

			Predicate pmmlPredicate = PredicateTranslator.translate(predicate, scope);

			Model model = estimator.encode(schema);

			Segment segment = new Segment(pmmlPredicate, model)
				.setId(name);

			segmentation.addSegments(segment);
		}

		MiningModel miningModel = new MiningModel(miningFunction, ModelUtil.createMiningSchema(label))
			.setSegmentation(segmentation);

		optimizeOutputFields(miningModel);

		return miningModel;
	}

	static
	public void optimizeOutputFields(MiningModel miningModel){
		Segmentation segmentation = miningModel.requireSegmentation();

		Map<String, OutputField> commonOutputFields = collectCommonOutputFields(segmentation);
		if(!commonOutputFields.isEmpty()){
			Output output = ModelUtil.ensureOutput(miningModel);

			removeCommonOutputFields(segmentation, commonOutputFields.keySet());

			List<OutputField> outputFields = output.getOutputFields();
			outputFields.addAll(commonOutputFields.values());
		}
	}

	static
	private Map<String, OutputField> collectCommonOutputFields(Segmentation segmentation){
		List<Segment> segments = segmentation.requireSegments();

		Map<String, OutputField> result = null;

		for(Segment segment : segments){
			Model model = segment.requireModel();

			Model finalModel = MiningModelUtil.getFinalModel(model);

			Output output = finalModel.getOutput();
			if(output != null && output.hasOutputFields()){
				List<OutputField> outputFields = output.getOutputFields();

				if(result == null){
					result = outputFields.stream()
						.filter((outputField) -> {
							ResultFeature resultFeature = outputField.getResultFeature();

							switch(resultFeature){
								case PROBABILITY:
								case AFFINITY:
									return true;
								default:
									return false;
							}
						})
						.collect(Collectors.toMap(outputField -> outputField.requireName(), outputField -> outputField));
				} else

				{
					Set<String> names = new LinkedHashSet<>();

					for(OutputField outputField : outputFields){
						String name = outputField.requireName();

						names.add(name);

						OutputField commonOutputField = result.get(name);
						if(commonOutputField != null && !ReflectionUtil.equals(outputField, commonOutputField)){
							result.remove(name);
						}
					}

					(result.keySet()).retainAll(names);
				}
			} else

			{
				result = Collections.emptyMap();
			} // End if

			if(result.isEmpty()){
				break;
			}
		}

		return result;
	}

	static
	private void removeCommonOutputFields(Segmentation segmentation, Set<String> names){
		List<Segment> segments = segmentation.requireSegments();

		for(Segment segment : segments){
			Model model = segment.requireModel();

			Model finalModel = MiningModelUtil.getFinalModel(model);

			Output output = finalModel.getOutput();
			if(output != null && output.hasOutputFields()){
				List<OutputField> outputFields = output.getOutputFields();

				outputFields.removeIf((outputField) -> {
					return names.contains(outputField.requireName());
				});

				if(outputFields.isEmpty()){
					finalModel.setOutput(null);
				}
			}
		}
	}
}