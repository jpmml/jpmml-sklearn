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
import org.dmg.pmml.Value;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.converter.Schema;
import org.jpmml.converter.ValueUtil;
import org.jpmml.sklearn.ClassDictUtil;
import org.jpmml.sklearn.FeatureMapper;
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

		FeatureMapper featureMapper = new FeatureMapper();

		DataField dataField = null;

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

			dataField = featureMapper.createDataField(PMMLPipeline.nameFunction.apply(targetField), opType, dataType);

			if(targetCategories != null && targetCategories.size() > 0){
				List<Value> values = dataField.getValues();

				values.addAll(PMMLUtil.createValues(targetCategories));
			}
		} // End if

		if(dataFrameMapper != null){
			dataFrameMapper.encodeFeatures(featureMapper);

			featureMapper.updateFeatures(getOpType(), getDataType());
		} else

		{
			List<String> activeFields = getActiveFields();

			if(activeFields == null){
				activeFields = new ArrayList<>();

				for(int i = 0, max = getNumberOfFeatures(); i < max; i++){
					activeFields.add("x" + String.valueOf(i + 1));
				}
			}

			featureMapper.initFeatures(Lists.transform(activeFields, PMMLPipeline.nameFunction), getOpType(), getDataType());
		}

		Schema schema;

		if(estimator.isSupervised()){
			schema = featureMapper.createSchema(dataField.getName(), PMMLUtil.getValues(dataField));
		} else

		{
			schema = featureMapper.createSchema(null, null);
		} // End if

		if(estimator.requiresContinuousInput()){
			schema = featureMapper.cast(OpType.CONTINUOUS, estimator.getDataType(), schema);
		}

		Set<DefineFunction> defineFunctions = encodeDefineFunctions();
		for(DefineFunction defineFunction : defineFunctions){
			featureMapper.addDefineFunction(defineFunction);
		}

		Model model = encodeModel(schema, featureMapper);

		return featureMapper.encodePMML(model);
	}

	public DataFrameMapper getMapper(){
		Object[] mapperStep = getMapperStep();

		if(mapperStep != null){
			return (DataFrameMapper)TupleUtil.extractElement(mapperStep, 1);
		}

		return null;
	}

	public Object[] getMapperStep(){
		List<Object[]> selectorSteps = super.getSelectorSteps();

		if(selectorSteps.size() > 0){
			Object object = TupleUtil.extractElement(selectorSteps.get(0), 1);

			if(object instanceof DataFrameMapper){
				return selectorSteps.get(0);
			}
		}

		return null;
	}

	@Override
	public List<Object[]> getSelectorSteps(){
		List<Object[]> selectorSteps = super.getSelectorSteps();

		if(selectorSteps.size() > 0){
			Object object = TupleUtil.extractElement(selectorSteps.get(0), 1);

			if(object instanceof DataFrameMapper){
				selectorSteps = selectorSteps.subList(1, selectorSteps.size());
			}
		}

		return selectorSteps;
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

	private static final Function<String, FieldName> nameFunction = new Function<String, FieldName>(){

		@Override
		public FieldName apply(String string){
			return FieldName.create(string);
		}
	};
}