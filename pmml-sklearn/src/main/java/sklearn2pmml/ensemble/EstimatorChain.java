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
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

import com.google.common.collect.Iterables;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.Model;
import org.dmg.pmml.Predicate;
import org.dmg.pmml.mining.MiningModel;
import org.dmg.pmml.mining.Segment;
import org.dmg.pmml.mining.Segmentation;
import org.jpmml.converter.Feature;
import org.jpmml.converter.Label;
import org.jpmml.converter.MultiLabel;
import org.jpmml.converter.Schema;
import org.jpmml.converter.mining.MiningModelUtil;
import org.jpmml.python.DataFrameScope;
import org.jpmml.python.PredicateTranslator;
import org.jpmml.python.Scope;
import org.jpmml.python.TupleUtil;
import sklearn.Estimator;
import sklearn.HasEstimatorEnsemble;

public class EstimatorChain extends Estimator implements HasEstimatorEnsemble<Estimator> {

	public EstimatorChain(String module, String name){
		super(module, name);
	}

	@Override
	public MiningFunction getMiningFunction(){
		List<? extends Estimator> estimators = getEstimators();

		Set<MiningFunction> miningFunctions = estimators.stream()
			.map(estimator -> estimator.getMiningFunction())
			.collect(Collectors.toSet());

		if(miningFunctions.size() == 1){
			return Iterables.getOnlyElement(miningFunctions);
		}

		return MiningFunction.MIXED;
	}

	@Override
	public boolean isSupervised(){
		return true;
	}

	@Override
	public Model encodeModel(Schema schema){
		MiningFunction miningFunction = getMiningFunction();
		List<Object[]> steps = getSteps();

		if(steps.isEmpty()){
			throw new IllegalArgumentException();
		}

		Label label = schema.getLabel();
		List<? extends Feature> features = schema.getFeatures();

		MultiLabel multiLabel = (MultiLabel)label;

		List<Model> models = new ArrayList<>();

		Segmentation segmentation = new Segmentation(Segmentation.MultipleModelMethod.MULTI_MODEL_CHAIN, null);

		Scope scope = new DataFrameScope("X", features);

		for(int i = 0; i < steps.size(); i++){
			Object[] step = steps.get(i);

			String name = TupleUtil.extractElement(step, 0, String.class);
			Estimator estimator = TupleUtil.extractElement(step, 1, Estimator.class);
			String predicate = TupleUtil.extractElement(step, 2, String.class);

			Schema segmentSchema = schema.toRelabeledSchema(multiLabel.getLabel(i));

			Predicate pmmlPredicate = PredicateTranslator.translate(predicate, scope);

			Model model = estimator.encode(segmentSchema);

			models.add(model);

			Segment segment = new Segment(pmmlPredicate, model)
				.setId(name);

			segmentation.addSegments(segment);
		}

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

	public List<Object[]> getSteps(){
		return getTupleList("steps");
	}
}