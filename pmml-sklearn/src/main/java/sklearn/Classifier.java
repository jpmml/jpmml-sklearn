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

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import numpy.core.ScalarUtil;
import org.dmg.pmml.DataField;
import org.dmg.pmml.DataType;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.Model;
import org.dmg.pmml.OpType;
import org.dmg.pmml.Output;
import org.dmg.pmml.OutputField;
import org.dmg.pmml.Value;
import org.dmg.pmml.Visitor;
import org.dmg.pmml.VisitorAction;
import org.jpmml.converter.CategoricalLabel;
import org.jpmml.converter.DiscreteLabel;
import org.jpmml.converter.ExceptionUtil;
import org.jpmml.converter.Label;
import org.jpmml.converter.MissingLabelException;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.MultiLabel;
import org.jpmml.converter.OrdinalLabel;
import org.jpmml.converter.TypeUtil;
import org.jpmml.converter.ValueUtil;
import org.jpmml.converter.mining.MiningModelUtil;
import org.jpmml.converter.visitors.AbstractExtender;
import org.jpmml.python.CastFunction;
import org.jpmml.python.ClassDictUtil;
import org.jpmml.python.HasArray;
import org.jpmml.sklearn.SkLearnEncoder;
import org.jpmml.sklearn.SkLearnException;
import sklearn2pmml.SkLearn2PMMLFields;

abstract
public class Classifier extends Estimator implements HasClasses {

	public Classifier(String module, String name){
		super(module, name);
	}

	@Override
	public MiningFunction getMiningFunction(){
		return MiningFunction.CLASSIFICATION;
	}

	@Override
	public boolean isSupervised(){
		return true;
	}

	@Override
	public int getNumberOfOutputs(){
		int numberOfOutputs = super.getNumberOfOutputs();

		if(numberOfOutputs == HasNumberOfOutputs.UNKNOWN){
			numberOfOutputs = 1;
		}

		return numberOfOutputs;
	}

	@Override
	public List<?> getClasses(){

		if(hasattr(SkLearn2PMMLFields.PMML_CLASSES)){
			return getClasses(SkLearn2PMMLFields.PMML_CLASSES);
		}

		return getClasses(SkLearnFields.CLASSES);
	}

	protected List<?> getClasses(String name){
		List<?> values = getListLike(name);

		values = values.stream()
			.map(value -> {

				if(value instanceof HasArray){
					HasArray hasArray = (HasArray)value;

					return canonicalizeValues(hasArray.getArrayContent());
				}

				return value;
			})
			.collect(Collectors.toList());

		return canonicalizeValues(values);
	}

	@Override
	public boolean hasProbabilityDistribution(){
		return true;
	}

	@Override
	public Label encodeLabel(List<String> names, SkLearnEncoder encoder){
		List<?> classes = getClasses();

		if(names.size() == 1){
			return encodeLabel(names.get(0), classes, encoder);
		} else

		if(names.size() >= 2){
			List<Label> labels = new ArrayList<>();

			for(int i = 0; i < names.size(); i++){
				String name = names.get(i);

				CastFunction<List<?>> castFunction = new CastFunction<List<?>>((Class)List.class){

					@Override
					public List<?> apply(Object object){

						try {
							return super.apply(object);
						} catch(ClassCastException cce){
							throw new SkLearnException("The categories object of the " + (name != null ? ExceptionUtil.formatName(name) : "<un-named> ") + " target field (" + ClassDictUtil.formatClass(object) + ") is not supported", cce);
						}
					}
				};

				List<?> categories = castFunction.apply(classes.get(i));

				Label label = encodeLabel(name, categories, encoder);

				labels.add(label);
			}

			return new MultiLabel(labels);
		} else

		{
			throw new MissingLabelException();
		}
	}

	protected DiscreteLabel encodeLabel(String name, List<?> categories, SkLearnEncoder encoder){
		DataType dataType = TypeUtil.getDataType(categories, DataType.STRING);

		return encodeLabel(name, OpType.CATEGORICAL, dataType, categories, encoder);
	}

	protected DiscreteLabel encodeLabel(String name, OpType opType, DataType dataType, List<?> categories, SkLearnEncoder encoder){

		if(name != null){
			DataField dataField = encoder.createDataField(name, opType, dataType, categories);

			Map<String, Map<String, ?>> classExtensions = (Map)getOption(HasClassifierOptions.OPTION_CLASS_EXTENSIONS, null);
			if(classExtensions != null){
				addClassExtensions(dataField, classExtensions);
			}

			switch(opType){
				case CATEGORICAL:
					return new CategoricalLabel(dataField);
				case ORDINAL:
					return new OrdinalLabel(dataField);
				default:
					throw new IllegalArgumentException();
			}
		} else

		{
			switch(opType){
				case CATEGORICAL:
					return new CategoricalLabel(dataType, categories);
				case ORDINAL:
					return new OrdinalLabel(dataType, categories);
				default:
					throw new IllegalArgumentException();
			}
		}
	}

	private void addClassExtensions(DataField dataField, Map<String, Map<String, ?>> classExtensions){
		List<Visitor> visitors = new ArrayList<>();

		if(classExtensions != null){
			Collection<? extends Map.Entry<String, Map<String, ?>>> entries = classExtensions.entrySet();

			for(Map.Entry<String, Map<String, ?>> entry : entries){
				String name = entry.getKey();

				Map<String, ?> values = entry.getValue();

				Visitor valueExtender = new AbstractExtender(name){

					@Override
					public VisitorAction visit(Value pmmlValue){
						Object value = values.get(pmmlValue.requireValue());

						if(value != null){
							value = ScalarUtil.decode(value);

							addExtension(pmmlValue, ValueUtil.asString(value));
						}

						return super.visit(pmmlValue);
					}
				};

				visitors.add(valueExtender);
			}
		}

		for(Visitor visitor : visitors){
			visitor.applyTo(dataField);
		}
	}

	public List<OutputField> encodePredictProbaOutput(Model model, DataType dataType, DiscreteLabel discreteLabel){
		List<OutputField> predictProbaFields = createPredictProbaFields(dataType, discreteLabel);

		model = MiningModelUtil.getFinalModel(model);

		Output output = ModelUtil.ensureOutput(model);

		(output.getOutputFields()).addAll(predictProbaFields);

		return predictProbaFields;
	}

	static
	public Object canonicalizeValue(Object value){

		if(value instanceof Long){
			Long longValue = (Long)value;

			return Math.toIntExact(longValue);
		}

		return value;
	}

	static
	public List<?> canonicalizeValues(List<?> values){
		return values.stream()
			.map(value -> canonicalizeValue(value))
			.collect(Collectors.toList());
	}

	public static final String FIELD_PROBABILITY = "probability";
}