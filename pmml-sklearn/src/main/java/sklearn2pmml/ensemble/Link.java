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

import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.Model;
import org.jpmml.converter.Feature;
import org.jpmml.converter.Label;
import org.jpmml.converter.Schema;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn.Estimator;
import sklearn.EstimatorUtil;
import sklearn.HasClasses;
import sklearn.HasEstimator;
import sklearn.SkLearnMethods;
import sklearn.tree.HasTreeOptions;

public class Link extends Estimator implements HasClasses, HasEstimator<Estimator> {

	public Link(String module, String name){
		super(module, name);
	}

	@Override
	public MiningFunction getMiningFunction(){
		Estimator estimator = getEstimator();

		return estimator.getMiningFunction();
	}

	@Override
	public String getAlgorithmName(){
		Estimator estimator = getEstimator();

		return estimator.getAlgorithmName();
	}

	@Override
	public List<?> getClasses(){
		Estimator estimator = getEstimator();

		return EstimatorUtil.getClasses(estimator);
	}

	@Override
	public Label encodeLabel(List<String> names, SkLearnEncoder encoder){
		Estimator estimator = getEstimator();

		return estimator.encodeLabel(names, encoder);
	}

	@Override
	public Model encodeModel(Schema schema){
		List<String> augmentFuncs = getAugmentFuncs();
		Estimator estimator = getEstimator();

		for(String augmentFunc : augmentFuncs){

			switch(augmentFunc){
				case SkLearnMethods.APPLY:
					{
						if(estimator instanceof HasTreeOptions){
							HasTreeOptions hasTreeOptions = (HasTreeOptions)estimator;

							// XXX
							estimator.putOption(HasTreeOptions.OPTION_WINNER_ID, Boolean.TRUE);
						}
					}
					break;
				default:
					break;
			}
		}

		return estimator.encodeModel(schema);
	}

	public Schema augmentSchema(Model model, Schema schema){
		List<String> augmentFuncs = getAugmentFuncs();
		Estimator estimator = getEstimator();

		if(augmentFuncs.isEmpty()){
			return schema;
		}

		SkLearnEncoder encoder = (SkLearnEncoder)schema.getEncoder();
		Label label = schema.getLabel();
		List<? extends Feature> features = schema.getFeatures();

		List<Feature> augmentedFeatures = new ArrayList<>(features);

		for(String augmentFunc : augmentFuncs){
			augmentedFeatures.addAll(EstimatorUtil.export(estimator, augmentFunc, schema, model, encoder));
		}

		return new Schema(encoder, label, augmentedFeatures);
	}


	public List<String> getAugmentFuncs(){
		return getList("augment_funcs", String.class);
	}

	@Override
	public Estimator getEstimator(){
		return get("estimator_", Estimator.class);
	}
}