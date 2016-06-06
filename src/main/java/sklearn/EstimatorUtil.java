/*
 * Copyright (c) 2015 Villu Ruusmann
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
package sklearn;

import java.util.Arrays;
import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

import com.google.common.base.Function;
import com.google.common.collect.Lists;
import org.dmg.pmml.DataField;
import org.dmg.pmml.DataType;
import org.dmg.pmml.DefineFunction;
import org.dmg.pmml.Expression;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.FieldRef;
import org.dmg.pmml.MiningField;
import org.dmg.pmml.MiningModel;
import org.dmg.pmml.MiningSchema;
import org.dmg.pmml.Model;
import org.dmg.pmml.OpType;
import org.dmg.pmml.Output;
import org.dmg.pmml.OutputField;
import org.dmg.pmml.PMML;
import org.dmg.pmml.PMMLObject;
import org.dmg.pmml.ParameterField;
import org.dmg.pmml.Segment;
import org.dmg.pmml.Segmentation;
import org.dmg.pmml.TransformationDictionary;
import org.dmg.pmml.Visitor;
import org.jpmml.converter.FeatureSchema;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.model.visitors.DictionaryCleaner;
import org.jpmml.model.visitors.MiningSchemaCleaner;
import org.jpmml.sklearn.ClassDictUtil;
import org.jpmml.sklearn.FeatureMapper;
import org.jpmml.sklearn.MiningFieldDecorator;

public class EstimatorUtil {

	private EstimatorUtil(){
	}

	static
	public PMML encodePMML(Estimator estimator, FeatureMapper featureMapper){
		FeatureSchema schema = estimator.createSchema(featureMapper);

		PMML pmml = featureMapper.encodePMML();

		Set<DefineFunction> defineFunctions = estimator.encodeDefineFunctions();
		if(defineFunctions != null && defineFunctions.size() > 0){
			TransformationDictionary transformationDictionary = pmml.getTransformationDictionary();

			if(transformationDictionary == null){
				transformationDictionary = new TransformationDictionary();

				pmml.setTransformationDictionary(transformationDictionary);
			}

			for(DefineFunction defineFunction : defineFunctions){
				transformationDictionary.addDefineFunctions(defineFunction);
			}
		}

		Model model = estimator.encodeModel(schema);

		pmml.addModels(model);

		// XXX
		MiningSchemaCleaner miningSchemaCleaner = new MiningSchemaCleaner(){

			@Override
			public PMMLObject popParent(){
				PMMLObject parent = super.popParent();

				if(parent instanceof MiningModel){
					cleanMiningSchema((MiningModel)parent);
				}

				return parent;
			}

			private void cleanMiningSchema(MiningModel miningModel){
				Set<FieldName> outputFieldNames = collectSegmentationOutputFields(miningModel);

				MiningSchema miningSchema = miningModel.getMiningSchema();

				List<MiningField> miningFields = miningSchema.getMiningFields();
				for(Iterator<MiningField> miningFieldIt = miningFields.iterator(); miningFieldIt.hasNext(); ){
					MiningField miningField = miningFieldIt.next();

					FieldName name = miningField.getName();
					if(outputFieldNames.contains(name)){
						miningFieldIt.remove();
					}
				}
			}

			private Set<FieldName> collectSegmentationOutputFields(MiningModel miningModel){
				Segmentation segmentation = miningModel.getSegmentation();

				Set<FieldName> names = new LinkedHashSet<>();

				List<Segment> segments = segmentation.getSegments();
				for(Segment segment : segments){
					Model model = segment.getModel();

					Output output = model.getOutput();
					if(output != null && output.hasOutputFields()){
						List<OutputField> outputFields = output.getOutputFields();

						for(OutputField outputField : outputFields){
							names.add(outputField.getName());
						}
					}
				}

				return names;
			}
		};

		DictionaryCleaner dictionaryCleaner = new DictionaryCleaner();

		List<? extends Visitor> visitors = Arrays.asList(miningSchemaCleaner, dictionaryCleaner);
		for(Visitor visitor : visitors){
			visitor.applyTo(pmml);
		}

		MiningSchema miningSchema = model.getMiningSchema();

		List<MiningField> miningFields = miningSchema.getMiningFields();
		for(MiningField miningField : miningFields){
			FieldName name = miningField.getName();

			List<MiningFieldDecorator> decorators = featureMapper.getDecorators(name);
			if(decorators == null){
				continue;
			}

			DataField dataField = (DataField)featureMapper.getField(name);

			for(MiningFieldDecorator decorator : decorators){
				decorator.decorate(dataField, miningField);
			}
		}

		return pmml;
	}

	static
	public FeatureSchema createSegmentSchema(FeatureSchema schema){
		FeatureSchema result = new FeatureSchema(null, schema.getTargetCategories(), schema.getActiveFields(), schema.getFeatures());

		return result;
	}

	static
	public Classifier asClassifier(Object object){
		return EstimatorUtil.classifierTransformer.apply(object);
	}

	static
	public List<? extends Classifier> asClassifierList(List<?> objects){
		return Lists.transform(objects, EstimatorUtil.classifierTransformer);
	}

	static
	public Regressor asRegressor(Object object){
		return EstimatorUtil.regressorTransformer.apply(object);
	}

	static
	public List<? extends Regressor> asRegressorList(List<?> objects){
		return Lists.transform(objects, EstimatorUtil.regressorTransformer);
	}

	static
	public DefineFunction encodeLogitFunction(){
		return encodeLossFunction("logit", -1d);
	}

	static
	public DefineFunction encodeAdaBoostFunction(){
		return encodeLossFunction("adaboost", -2d);
	}

	static
	private DefineFunction encodeLossFunction(String function, double multiplier){
		FieldName name = FieldName.create("value");

		ParameterField parameterField = new ParameterField(name)
			.setDataType(DataType.DOUBLE)
			.setOpType(OpType.CONTINUOUS);

		// "1 / (1 + exp($multiplier * $name))"
		Expression expression = PMMLUtil.createApply("/", PMMLUtil.createConstant(1d), PMMLUtil.createApply("+", PMMLUtil.createConstant(1d), PMMLUtil.createApply("exp", PMMLUtil.createApply("*", PMMLUtil.createConstant(multiplier), new FieldRef(name)))));

		DefineFunction defineFunction = new DefineFunction(function, OpType.CONTINUOUS, null)
			.setDataType(DataType.DOUBLE)
			.setOpType(OpType.CONTINUOUS)
			.addParameterFields(parameterField)
			.setExpression(expression);

		return defineFunction;
	}

	private static final Function<Object, Classifier> classifierTransformer = new Function<Object, Classifier>(){

		@Override
		public Classifier apply(Object object){

			try {
				if(object == null){
					throw new NullPointerException();
				}

				return (Classifier)object;
			} catch(RuntimeException re){
				throw new IllegalArgumentException("The estimator object (" + ClassDictUtil.formatClass(object) + ") is not a Classifier or is not a supported Classifier subclass", re);
			}
		}
	};

	private static final Function<Object, Regressor> regressorTransformer = new Function<Object, Regressor>(){

		@Override
		public Regressor apply(Object object){

			try {
				if(object == null){
					throw new NullPointerException();
				}

				return (Regressor)object;
			} catch(RuntimeException re){
				throw new IllegalArgumentException("The estimator object (" + ClassDictUtil.formatClass(object) + ") is not a Regressor or is not a supported Regressor subclass", re);
			}
		}
	};
}