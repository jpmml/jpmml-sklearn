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
package sklearn_pandas;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

import com.google.common.base.Function;
import com.google.common.collect.Lists;
import com.google.common.collect.Sets;
import net.razorvine.pickle.objects.ClassDict;
import org.dmg.pmml.DataField;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.NormDiscrete;
import org.jpmml.converter.BinaryFeature;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.Feature;
import org.jpmml.converter.PseudoFeature;
import org.jpmml.sklearn.ClassDictUtil;
import org.jpmml.sklearn.FeatureMapper;
import org.jpmml.sklearn.TupleUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import sklearn.Transformer;

public class DataFrameMapper extends ClassDict {

	public DataFrameMapper(String module, String name){
		super(module, name);
	}

	public void encodeFeatures(FeatureMapper featureMapper){
		List<Object[]> steps = getFeatures();

		if(steps.size() < 1){
			logger.warn("The list of mappings is empty");

			return;
		}

		Set<FieldName> mappedNames = new LinkedHashSet<>();

		for(int row = 0; row < steps.size(); row++){
			Object[] step = steps.get(row);

			List<Feature> features = new ArrayList<>();

			Set<FieldName> uniqueNames = new LinkedHashSet<>();
			Set<FieldName> duplicateNames = new LinkedHashSet<>();

			List<FieldName> names = getNameList(step);
			for(FieldName name : names){
				DataField dataField = featureMapper.createDataField(name);

				Feature feature = new PseudoFeature(dataField);

				features.add(feature);

				boolean unique = uniqueNames.add(name);
				if(!unique){
					duplicateNames.add(name);
				}
			}

			Sets.SetView<FieldName> duplicateMappedNames = Sets.intersection(uniqueNames, mappedNames);
			if(duplicateMappedNames.size() > 0){
				duplicateMappedNames.copyInto(duplicateNames);
			} // End if

			if(duplicateNames.size() > 0){
				logger.error("Duplicate mappings(s): {}", new ArrayList<>(duplicateNames));

				throw new IllegalArgumentException();
			}

			List<Transformer> transformers = getTransformerList(step);
			for(int column = 0; column < transformers.size(); column++){
				Transformer transformer = transformers.get(column);

				for(Feature feature : features){
					featureMapper.updateType(feature.getName(), transformer.getOpType(), transformer.getDataType());
				}

				features = transformer.encodeFeatures("(" + row + "," + column + ")", features, featureMapper);
			} // End for

			for(int i = 0; i < features.size(); i++){
				Feature feature = features.get(i);

				if(feature instanceof BinaryFeature){
					BinaryFeature binaryFeature = (BinaryFeature)feature;

					NormDiscrete normDiscrete = new NormDiscrete(binaryFeature.getName(), binaryFeature.getValue());

					DerivedField derivedField = featureMapper.createDerivedField(FieldName.create((binaryFeature.getName()).getValue() + "[" + binaryFeature.getValue() + "]"), normDiscrete);

					features.set(i, new ContinuousFeature(derivedField));
				}
			}

			featureMapper.addStep(features);
		}
	}

	public List<Object[]> getFeatures(){
		return (List)get("features");
	}

	static
	private List<FieldName> getNameList(Object[] feature){
		Function<Object, FieldName> function = new Function<Object, FieldName>(){

			@Override
			public FieldName apply(Object object){

				if(object instanceof String){
					return FieldName.create((String)object);
				}

				throw new IllegalArgumentException("The key object (" + ClassDictUtil.formatClass(object) + ") is not a String");
			}
		};

		try {
			if(feature[0] instanceof List){
				return new ArrayList<>(Lists.transform(((List)feature[0]), function));
			}

			return Collections.singletonList(function.apply(feature[0]));
		} catch(RuntimeException re){
			throw new IllegalArgumentException("Invalid mapping key", re);
		}
	}

	static
	private List<Transformer> getTransformerList(Object[] feature){
		Function<Object, Transformer> function = new Function<Object, Transformer>(){

			@Override
			public Transformer apply(Object object){

				if(object instanceof Transformer){
					return (Transformer)object;
				}

				throw new IllegalArgumentException("The value object (" + ClassDictUtil.formatClass(object) + ") is not a Transformer or is not a supported Transformer subclass");
			}
		};

		try {
			if(feature[1] == null){
				return Collections.emptyList();
			} // End if

			if(feature[1] instanceof TransformerPipeline){
				TransformerPipeline transformerPipeline = (TransformerPipeline)feature[1];

				List<Object[]> steps = transformerPipeline.getSteps();

				return new ArrayList<>(Lists.transform((List)TupleUtil.extractElement(steps, 1), function));
			} // End if

			if(feature[1] instanceof List){
				return new ArrayList<>(Lists.transform((List)feature[1], function));
			}

			return Collections.singletonList(function.apply(feature[1]));
		} catch(RuntimeException re){
			throw new IllegalArgumentException("Invalid mapping value", re);
		}
	}

	private static final Logger logger = LoggerFactory.getLogger(DataFrameMapper.class);
}