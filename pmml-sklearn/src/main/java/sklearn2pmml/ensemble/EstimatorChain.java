/*
 * Copyright (c) 2022 Villu Ruusmann
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
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.Model;
import org.dmg.pmml.Predicate;
import org.dmg.pmml.mining.MiningModel;
import org.dmg.pmml.mining.Segment;
import org.dmg.pmml.mining.Segmentation;
import org.jpmml.converter.Feature;
import org.jpmml.converter.Label;
import org.jpmml.converter.MultiLabel;
import org.jpmml.converter.ScalarLabel;
import org.jpmml.converter.ScalarLabelUtil;
import org.jpmml.converter.Schema;
import org.jpmml.converter.mining.MiningModelUtil;
import org.jpmml.python.ClassDictUtil;
import org.jpmml.python.DataFrameScope;
import org.jpmml.python.Scope;
import org.jpmml.python.TupleUtil;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn.Estimator;
import sklearn.EstimatorUtil;
import sklearn.HasClasses;
import sklearn.HasEstimatorEnsemble;
import sklearn2pmml.util.EvaluatableUtil;

public class EstimatorChain extends Estimator implements HasClasses, HasEstimatorEnsemble<Estimator> {

	public EstimatorChain(String module, String name){
		super(module, name);
	}

	@Override
	public MiningFunction getMiningFunction(){
		List<? extends Estimator> estimators = getEstimators();

		return EstimatorUtil.getMiningFunction(estimators);
	}

	@Override
	public int getNumberOfOutputs(){
		Boolean multioutput = getMultioutput();

		if(multioutput){
			List<? extends Estimator> estimators = getEstimators();

			return estimators.size();
		}

		return 1;
	}

	@Override
	public boolean isSupervised(){
		return true;
	}

	@Override
	public List<?> getClasses(){
		List<? extends Estimator> estimators = getEstimators();

		if(estimators.size() == 1){
			Estimator estimator = estimators.get(0);

			return EstimatorUtil.getClasses(estimator);
		} else

		if(estimators.size() >= 2){
			List<Object> result = new ArrayList<>();

			for(Estimator estimator : estimators){
				result.add(EstimatorUtil.getClasses(estimator));
			}

			List<Object> uniqueResults = result.stream()
				.distinct()
				.collect(Collectors.toList());

			if(uniqueResults.size() == 1){
				return (List<?>)uniqueResults.get(0);
			}

			return result;
		} else

		{
			throw new IllegalArgumentException();
		}
	}

	@Override
	public Label encodeLabel(List<String> names, SkLearnEncoder encoder){
		List<? extends Estimator> estimators = getEstimators();

		ClassDictUtil.checkSize(names, estimators);

		if(names.size() == 1){
			String name = names.get(0);
			Estimator estimator = estimators.get(0);

			return estimator.encodeLabel(Collections.singletonList(name), encoder);
		} else

		if(names.size() >= 2){
			List<Label> labels = new ArrayList<>();

			for(int i = 0; i < names.size(); i++){
				String name = names.get(i);
				Estimator estimator = estimators.get(i);

				ScalarLabel label = (ScalarLabel)estimator.encodeLabel(Collections.singletonList(name), encoder);

				labels.add(label);
			}

			return new MultiLabel(labels);
		} else

		{
			throw new IllegalArgumentException();
		}
	}

	@Override
	public Model encodeModel(Schema schema){
		Boolean multioutput = getMultioutput();
		List<Object[]> steps = getSteps();

		if(steps.isEmpty()){
			throw new IllegalArgumentException();
		}

		Label label = schema.getLabel();
		List<? extends Feature> features = schema.getFeatures();

		List<ScalarLabel> scalarLabels = ScalarLabelUtil.toScalarLabels(label);

		if(multioutput){
			ClassDictUtil.checkSize(steps, scalarLabels);
		}

		List<Estimator> estimators = new ArrayList<>();

		List<Model> models = new ArrayList<>();

		Segmentation segmentation = new Segmentation(multioutput ? Segmentation.MultipleModelMethod.MULTI_MODEL_CHAIN : Segmentation.MultipleModelMethod.MODEL_CHAIN, null);

		Scope scope = new DataFrameScope("X", features);

		for(int i = 0; i < steps.size(); i++){
			Object[] step = steps.get(i);
			ScalarLabel scalarLabel = (multioutput ? scalarLabels.get(i) : scalarLabels.get(0));

			String name = TupleUtil.extractElement(step, 0, String.class);
			Estimator estimator = TupleUtil.extractElement(step, 1, Estimator.class);
			Object expr = TupleUtil.extractElement(step, 2, Object.class);

			estimators.add(estimator);

			Schema segmentSchema = schema.toRelabeledSchema(scalarLabel);

			Predicate predicate = EvaluatableUtil.translatePredicate(expr, scope);

			Model model;

			if(multioutput){
				model = estimator.encode(scalarLabel.getName(), segmentSchema);
			} else

			{
				model = estimator.encode(segmentSchema);
			}

			models.add(model);

			if(estimator instanceof Link){
				Link link = (Link)estimator;

				schema = link.augmentSchema(model, segmentSchema);
			}

			Segment segment = new Segment(predicate, model)
				.setId(name);

			segmentation.addSegments(segment);
		}

		MiningFunction miningFunction = EstimatorUtil.getMiningFunction(estimators);

		MiningModel miningModel = new MiningModel(miningFunction, MiningModelUtil.createMiningSchema(models))
			.setSegmentation(segmentation);

		return miningModel;
	}

	@Override
	public List<? extends Estimator> getEstimators(){
		List<Object[]> steps = getSteps();

		if(steps.isEmpty()){
			throw new IllegalArgumentException();
		}

		return TupleUtil.extractElementList(steps, 1, Estimator.class);
	}

	public Boolean getMultioutput(){
		return getBoolean("multioutput");
	}

	public List<Object[]> getSteps(){
		return getTupleList("steps");
	}
}