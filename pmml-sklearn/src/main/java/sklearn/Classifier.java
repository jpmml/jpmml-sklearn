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
import org.dmg.pmml.mining.MiningModel;
import org.jpmml.converter.CategoricalLabel;
import org.jpmml.converter.Label;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.MultiLabel;
import org.jpmml.converter.ScalarLabel;
import org.jpmml.converter.TypeUtil;
import org.jpmml.converter.ValueUtil;
import org.jpmml.converter.mining.MiningModelUtil;
import org.jpmml.converter.visitors.AbstractExtender;
import org.jpmml.python.CastFunction;
import org.jpmml.python.ClassDictUtil;
import org.jpmml.python.HasArray;
import org.jpmml.sklearn.SkLearnEncoder;

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
	public int getNumberOfOutputs(){
		int numberOfOutputs = super.getNumberOfOutputs();

		if(numberOfOutputs == HasNumberOfOutputs.UNKNOWN){
			numberOfOutputs = 1;
		}

		return numberOfOutputs;
	}

	@Override
	public List<?> getClasses(){
		List<?> values = getListLike(SkLearnFields.CLASSES);

		values = values.stream()
			.map(value -> (value instanceof HasArray) ? canonicalizeValues(((HasArray)value).getArrayContent()) : value)
			.collect(Collectors.toList());

		return canonicalizeValues(values);
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
					public String formatMessage(Object object){
						return "The categories object of the \'" + name + "\' target field (" + ClassDictUtil.formatClass(object) + ") is not supported";
					}
				};

				List<?> categories = castFunction.apply(classes.get(i));

				Label label = encodeLabel(name, categories, encoder);

				labels.add(label);
			}

			return new MultiLabel(labels);
		} else

		{
			throw new IllegalArgumentException();
		}
	}

	protected ScalarLabel encodeLabel(String name, List<?> categories, SkLearnEncoder encoder){
		DataType dataType = TypeUtil.getDataType(categories, DataType.STRING);

		DataField dataField = encoder.createDataField(name, OpType.CATEGORICAL, dataType, categories);

		Map<String, Map<String, ?>> classExtensions = (Map)getOption(HasClassifierOptions.OPTION_CLASS_EXTENSIONS, null);
		if(classExtensions != null){
			addClassExtensions(dataField, classExtensions);
		}

		return new CategoricalLabel(dataField);
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

	public boolean hasProbabilityDistribution(){
		return true;
	}

	public List<OutputField> encodePredictProbaOutput(Model model, DataType dataType, CategoricalLabel categoricalLabel){
		List<OutputField> predictProbaFields = createPredictProbaFields(dataType, categoricalLabel);

		if(model instanceof MiningModel){
			MiningModel miningModel = (MiningModel)model;

			model = MiningModelUtil.getFinalModel(miningModel);
		}

		Output output = ModelUtil.ensureOutput(model);

		(output.getOutputFields()).addAll(predictProbaFields);

		return predictProbaFields;
	}

	static
	private List<?> canonicalizeValues(List<?> values){
		return values.stream()
			.map(value -> (value instanceof Long) ? Math.toIntExact((Long)value) : value)
			.collect(Collectors.toList());
	}

	public static final String FIELD_PROBABILITY = "probability";
}