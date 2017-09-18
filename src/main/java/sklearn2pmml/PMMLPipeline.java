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
package sklearn2pmml;

import java.util.ArrayList;
import java.util.List;

import com.google.common.base.CharMatcher;
import com.google.common.base.Function;
import com.google.common.collect.Lists;
import net.razorvine.pickle.objects.ClassDict;
import numpy.core.NDArray;
import org.dmg.pmml.DataField;
import org.dmg.pmml.DataType;
import org.dmg.pmml.Extension;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.MiningBuildTask;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.Model;
import org.dmg.pmml.OpType;
import org.dmg.pmml.PMML;
import org.jpmml.converter.CategoricalLabel;
import org.jpmml.converter.ContinuousLabel;
import org.jpmml.converter.Feature;
import org.jpmml.converter.Label;
import org.jpmml.converter.Schema;
import org.jpmml.converter.ValueUtil;
import org.jpmml.converter.WildcardFeature;
import org.jpmml.sklearn.ClassDictUtil;
import org.jpmml.sklearn.SkLearnEncoder;
import org.jpmml.sklearn.TupleUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import sklearn.Estimator;
import sklearn.EstimatorUtil;
import sklearn.HasNumberOfFeatures;
import sklearn.Initializer;
import sklearn.Transformer;
import sklearn.TransformerUtil;
import sklearn.TypeUtil;
import sklearn.pipeline.Pipeline;

public class PMMLPipeline extends Pipeline {

	public PMMLPipeline(){
		this("sklearn2pmml", "PMMLPipeline");
	}

	public PMMLPipeline(String module, String name){
		super(module, name);
	}

	public PMML encodePMML(){
		List<Transformer> transformers = getTransformers();
		Estimator estimator = getEstimator();
		String repr = getRepr();

		SkLearnEncoder encoder = new SkLearnEncoder();

		Label label = null;

		if(estimator.isSupervised()){
			String targetField = getTargetField();

			if(targetField == null){
				targetField = "y";

				logger.warn("The 'target_field' attribute is not set. Assuming {} as the name of the target field", targetField);
			}

			MiningFunction miningFunction = estimator.getMiningFunction();
			switch(miningFunction){
				case CLASSIFICATION:
					{
						List<?> classes = EstimatorUtil.getClasses(estimator);

						DataField dataField = encoder.createDataField(FieldName.create(targetField), OpType.CATEGORICAL, TypeUtil.getDataType(classes, DataType.STRING), formatTargetCategories(classes));

						label = new CategoricalLabel(dataField);
					}
					break;
				case REGRESSION:
					{
						DataField dataField = encoder.createDataField(FieldName.create(targetField), OpType.CONTINUOUS, DataType.DOUBLE);

						label = new ContinuousLabel(dataField);
					}
					break;
				default:
					throw new IllegalArgumentException();
			}
		}

		List<Feature> features = new ArrayList<>();

		Transformer transformer = TransformerUtil.getHead(transformers);
		if(transformer != null){

			if(!(transformer instanceof Initializer)){
				features = initFeatures(transformer, transformer.getOpType(), transformer.getDataType(), encoder);
			}

			features = encodeFeatures(features, encoder);
		} else

		{
			features = initFeatures(estimator, estimator.getOpType(), estimator.getDataType(), encoder);
		}

		int numberOfFeatures = estimator.getNumberOfFeatures();
		if(numberOfFeatures > -1){
			ClassDictUtil.checkSize(numberOfFeatures, features);
		}

		Schema schema = new Schema(label, features);

		Model model = estimator.encodeModel(schema, encoder);

		PMML pmml = encoder.encodePMML(model);

		if(repr != null){
			Extension extension = new Extension()
				.addContent(repr);

			MiningBuildTask miningBuildTask = new MiningBuildTask()
				.addExtensions(extension);

			pmml.setMiningBuildTask(miningBuildTask);
		}

		return pmml;
	}

	private List<Feature> initFeatures(ClassDict object, OpType opType, DataType dataType, SkLearnEncoder encoder){
		List<String> activeFields = getActiveFields();

		if(activeFields == null){
			int numberOfFeatures = -1;

			if(object instanceof HasNumberOfFeatures){
				HasNumberOfFeatures hasNumberOfFeatures = (HasNumberOfFeatures)object;

				numberOfFeatures = hasNumberOfFeatures.getNumberOfFeatures();
			} // End if

			if(numberOfFeatures < 0){
				throw new IllegalArgumentException("The first transformer or estimator object (" + ClassDictUtil.formatClass(object) + ") does not specify the number of input features");
			}

			activeFields = new ArrayList<>(numberOfFeatures);

			for(int i = 0, max = numberOfFeatures; i < max; i++){
				activeFields.add("x" + String.valueOf(i + 1));
			}

			logger.warn("The 'active_fields' attribute is not set. Assuming {} as the names of active fields", activeFields);
		}

		List<Feature> result = new ArrayList<>();

		for(String activeField : activeFields){
			DataField dataField = encoder.createDataField(FieldName.create(activeField), opType, dataType);

			result.add(new WildcardFeature(encoder, dataField));
		}

		return result;
	}

	@Override
	public List<Transformer> getTransformers(){
		List<Object[]> steps = getSteps();

		if(steps.size() > 0){
			steps = steps.subList(0, steps.size() - 1);
		}

		return TransformerUtil.asTransformerList(TupleUtil.extractElementList(steps, 1));
	}

	public Estimator getEstimator(){
		List<Object[]> steps = getSteps();

		if(steps.size() < 1){
			throw new IllegalArgumentException("Expected one or more elements, got zero elements");
		}

		Object[] lastStep = steps.get(steps.size() - 1);

		return EstimatorUtil.asEstimator(TupleUtil.extractElement(lastStep, 1));
	}

	@Override
	public List<Object[]> getSteps(){
		return super.getSteps();
	}

	public PMMLPipeline setSteps(List<Object[]> steps){
		put("steps", steps);

		return this;
	}

	public String getRepr(){
		return (String)get("repr_");
	}

	public PMMLPipeline setRepr(String repr){
		put("repr_", repr);

		return this;
	}

	public String getTargetField(){
		return (String)get("target_field");
	}

	public PMMLPipeline setTargetField(String targetField){
		put("target_field", targetField);

		return this;
	}

	public List<String> getActiveFields(){

		if(!containsKey("active_fields")){
			return null;
		}

		return (List)ClassDictUtil.getArray(this, "active_fields");
	}

	public PMMLPipeline setActiveFields(List<String> activeFields){
		NDArray array = new NDArray();
		array.put("data", activeFields);
		array.put("fortran_order", Boolean.FALSE);

		put("active_fields", array);

		return this;
	}

	static
	private List<String> formatTargetCategories(List<?> objects){
		Function<Object, String> function = new Function<Object, String>(){

			@Override
			public String apply(Object object){
				String targetCategory = ValueUtil.formatValue(object);

				if(targetCategory == null || CharMatcher.WHITESPACE.matchesAnyOf(targetCategory)){
					throw new IllegalArgumentException(targetCategory);
				}

				return targetCategory;
			}
		};

		return Lists.transform(objects, function);
	}

	private static final Logger logger = LoggerFactory.getLogger(PMMLPipeline.class);
}