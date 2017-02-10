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
import java.util.Collections;
import java.util.List;

import com.google.common.base.CharMatcher;
import com.google.common.base.Function;
import com.google.common.collect.Lists;
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
import sklearn.Estimator;
import sklearn.EstimatorUtil;
import sklearn.TypeUtil;
import sklearn.pipeline.Pipeline;
import sklearn_pandas.DataFrameMapper;

public class PMMLPipeline extends Pipeline {

	public PMMLPipeline(){
		super("sklearn2pmml", "PMMLPipeline");
	}

	public PMMLPipeline(String module, String name){
		super(module, name);
	}

	public PMML encodePMML(){
		DataFrameMapper dataFrameMapper = getMapper();
		Estimator estimator = getEstimator();
		String repr = getRepr();

		while(estimator instanceof Pipeline){
			Pipeline pipeline = (Pipeline)estimator;

			estimator = pipeline.getEstimator();
		}

		SkLearnEncoder encoder = new SkLearnEncoder();

		Label label = null;

		if(estimator.isSupervised()){
			String targetField = getTargetField();

			if(targetField == null){
				targetField = "y";
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
		} // End if

		if(dataFrameMapper != null){
			dataFrameMapper.encodeFeatures(encoder);
		} else

		{
			List<String> activeFields = getActiveFields();

			if(activeFields == null){
				activeFields = new ArrayList<>();

				for(int i = 0, max = getNumberOfFeatures(); i < max; i++){
					activeFields.add("x" + String.valueOf(i + 1));
				}
			}

			OpType opType = getOpType();
			DataType dataType = getDataType();

			for(String activeField : activeFields){
				DataField dataField = encoder.createDataField(FieldName.create(activeField), opType, dataType);

				encoder.addRow(Collections.singletonList(activeField), Collections.<Feature>singletonList(new WildcardFeature(encoder, dataField)));
			}
		}

		Schema schema = new Schema(label, encoder.getFeatures());

		Model model = encodeModel(schema, encoder);

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

	public DataFrameMapper getMapper(){
		Object[] mapperStep = getMapperStep();

		if(mapperStep != null){
			return (DataFrameMapper)TupleUtil.extractElement(mapperStep, 1);
		}

		return null;
	}

	public Object[] getMapperStep(){
		List<Object[]> transformerSteps = super.getTransformerSteps();

		if(transformerSteps.size() > 0){
			Object object = TupleUtil.extractElement(transformerSteps.get(0), 1);

			if(object instanceof DataFrameMapper){
				return transformerSteps.get(0);
			}
		}

		return null;
	}

	@Override
	public List<Object[]> getTransformerSteps(){
		List<Object[]> transformerSteps = super.getTransformerSteps();

		if(transformerSteps.size() > 0){
			Object object = TupleUtil.extractElement(transformerSteps.get(0), 1);

			if(object instanceof DataFrameMapper){
				transformerSteps = transformerSteps.subList(1, transformerSteps.size());
			}
		}

		return transformerSteps;
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

	public String getTargetField(){
		return (String)get("target_field");
	}

	public PMMLPipeline setTargetField(String targetField){
		put("target_field", targetField);

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

		return new ArrayList<>(Lists.transform(objects, function));
	}
}