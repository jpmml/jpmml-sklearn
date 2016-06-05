/*
 * Copyright (c) 2016 Villu Ruusmann
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
package sklearn.ensemble.voting_classifier;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.dmg.pmml.DataType;
import org.dmg.pmml.DefineFunction;
import org.dmg.pmml.MiningFunctionType;
import org.dmg.pmml.MiningModel;
import org.dmg.pmml.MiningSchema;
import org.dmg.pmml.Model;
import org.dmg.pmml.MultipleModelMethodType;
import org.dmg.pmml.OpType;
import org.dmg.pmml.Output;
import org.dmg.pmml.Segmentation;
import org.jpmml.converter.FeatureSchema;
import org.jpmml.converter.MiningModelUtil;
import org.jpmml.converter.ModelUtil;
import org.jpmml.sklearn.ClassDictUtil;
import org.jpmml.sklearn.FeatureMapper;
import sklearn.Classifier;
import sklearn.Estimator;
import sklearn.EstimatorUtil;

public class VotingClassifier extends Classifier {

	private Map<List<?>, FeatureSchema> schemas = new HashMap<>();


	public VotingClassifier(String module, String name){
		super(module, name);
	}

	@Override
	public FeatureSchema createSchema(FeatureMapper featureMapper){
		FeatureSchema schema = super.createSchema(featureMapper);

		this.schemas.put(createSchemaKey(this), schema);

		List<? extends Classifier> estimators = getEstimators();
		for(Classifier estimator : estimators){
			List<?> schemaKey = createSchemaKey(estimator);

			FeatureSchema estimatorSchema = this.schemas.get(schemaKey);
			if(estimatorSchema == null){
				estimatorSchema = featureMapper.cast((OpType)schemaKey.get(0), (DataType)schemaKey.get(1), schema);

				this.schemas.put(schemaKey, estimatorSchema);
			}
		}

		return schema;
	}

	@Override
	public int getNumberOfFeatures(){
		List<? extends Classifier> estimators = getEstimators();

		Estimator estimator = estimators.get(0);

		return estimator.getNumberOfFeatures();
	}

	@Override
	public boolean requiresContinuousInput(){
		return false;
	}

	@Override
	public Set<DefineFunction> encodeDefineFunctions(){
		Map<String, DefineFunction> uniqueDefineFunctions = new LinkedHashMap<>();

		List<? extends Classifier> estimators = getEstimators();
		for(Classifier estimator : estimators){
			Set<DefineFunction> defineFunctions = estimator.encodeDefineFunctions();

			for(DefineFunction defineFunction : defineFunctions){
				uniqueDefineFunctions.put(defineFunction.getName(), defineFunction);
			}
		}

		Set<DefineFunction> result = new LinkedHashSet<>(uniqueDefineFunctions.values());

		return result;
	}

	@Override
	public Model encodeModel(FeatureSchema schema){
		List<? extends Classifier> estimators = getEstimators();
		List<? extends Number> weights = getWeights();

		List<Model> models = new ArrayList<>();

		for(Classifier estimator : estimators){
			List<?> schemaKey = createSchemaKey(estimator);

			FeatureSchema estimatorSchema = this.schemas.get(schemaKey);
			if(estimatorSchema == null){
				throw new IllegalArgumentException();
			}

			Model model = estimator.encodeModel(estimatorSchema);

			models.add(model);
		}

		String voting = getVoting();

		MultipleModelMethodType multipleModelMethod = encodeMultipleModelMethod(voting, (weights != null && weights.size() > 0));

		Segmentation segmentation = MiningModelUtil.createSegmentation(multipleModelMethod, models, weights);

		MiningSchema miningSchema = ModelUtil.createMiningSchema(schema);

		Output output = ModelUtil.createProbabilityOutput(schema);

		MiningModel miningModel = new MiningModel(MiningFunctionType.CLASSIFICATION, miningSchema)
			.setSegmentation(segmentation)
			.setOutput(output);

		return miningModel;
	}

	public List<? extends Classifier> getEstimators(){
		List<?> estimators = (List)get("estimators_");

		return EstimatorUtil.asClassifierList(estimators);
	}

	public String getVoting(){
		return (String)get("voting");
	}

	public List<? extends Number> getWeights(){
		Object weights = get("weights");

		if((weights == null) || (weights instanceof List)){
			return (List)weights;
		}

		return (List)ClassDictUtil.getArray(this, "weights");
	}

	static
	private MultipleModelMethodType encodeMultipleModelMethod(String voting, boolean weighted){

		if(("soft").equals(voting)){
			return (weighted ? MultipleModelMethodType.WEIGHTED_AVERAGE : MultipleModelMethodType.AVERAGE);
		} else

		if(("hard").equals(voting)){
			return (weighted ? MultipleModelMethodType.WEIGHTED_MAJORITY_VOTE : MultipleModelMethodType.MAJORITY_VOTE);
		} else

		{
			throw new IllegalArgumentException(voting);
		}
	}

	static
	private List<?> createSchemaKey(Estimator estimator){
		List<?> result = Arrays.asList(estimator.requiresContinuousInput() ? OpType.CONTINUOUS : null, estimator.getDataType());

		return result;
	}
}