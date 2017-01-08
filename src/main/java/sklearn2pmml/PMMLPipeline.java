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
import java.util.Set;

import com.google.common.base.CharMatcher;
import com.google.common.base.Function;
import com.google.common.collect.Lists;
import org.dmg.pmml.DataField;
import org.dmg.pmml.DataType;
import org.dmg.pmml.DefineFunction;
import org.dmg.pmml.FieldName;
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
import sklearn.Classifier;
import sklearn.Estimator;
import sklearn.TypeUtil;
import sklearn.pipeline.Pipeline;
import sklearn_pandas.DataFrameMapper;

public class PMMLPipeline extends Pipeline {

	public PMMLPipeline(String module, String name){
		super(module, name);
	}

	public PMML encodePMML(){
		DataFrameMapper dataFrameMapper = getMapper();
		Estimator estimator = getEstimator();

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

			OpType opType = OpType.CONTINUOUS;
			DataType dataType = DataType.DOUBLE;

			List<String> targetCategories = null;

			if(estimator instanceof Classifier){
				Classifier classifier = (Classifier)estimator;

				List<?> classes = classifier.getClasses();
				if(classes == null || classes.isEmpty()){
					throw new IllegalArgumentException();
				}

				opType = OpType.CATEGORICAL;
				dataType = TypeUtil.getDataType(classes, DataType.STRING);

				targetCategories = formatTargetCategories(classes);
			}

			DataField dataField = encoder.createDataField(FieldName.create(targetField), opType, dataType, targetCategories);

			if(targetCategories != null && targetCategories.size() > 0){
				label = new CategoricalLabel(dataField);
			} else

			{
				label = new ContinuousLabel(dataField);
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

		Set<DefineFunction> defineFunctions = encodeDefineFunctions();
		for(DefineFunction defineFunction : defineFunctions){
			encoder.addDefineFunction(defineFunction);
		}

		Schema schema = new Schema(label, encoder.getFeatures());

		Model model = encodeModel(schema, encoder);

		return encoder.encodePMML(model);
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

	public List<String> getActiveFields(){
		return (List)ClassDictUtil.getArray(this, "active_fields");
	}

	public String getTargetField(){
		return (String)get("target_field");
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