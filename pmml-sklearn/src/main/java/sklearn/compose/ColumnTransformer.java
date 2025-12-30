/*
 * Copyright (c) 2019 Villu Ruusmann
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
package sklearn.compose;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

import org.dmg.pmml.PMML;
import org.jpmml.converter.Feature;
import org.jpmml.python.CastFunction;
import org.jpmml.python.HasArray;
import org.jpmml.python.TupleUtil;
import org.jpmml.sklearn.Encodable;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn.Drop;
import sklearn.HasFeatureNamesIn;
import sklearn.HasSparseOutput;
import sklearn.Initializer;
import sklearn.InitializerUtil;
import sklearn.PassThrough;
import sklearn.SkLearnSteps;
import sklearn.Transformer;
import sklearn.TransformerCastFunction;
import sklearn.TransformerUtil;

public class ColumnTransformer extends Initializer implements HasFeatureNamesIn, HasSparseOutput, Encodable {

	public ColumnTransformer(String module, String name){
		super(module, name);
	}

	@Override
	public List<Feature> initializeFeatures(SkLearnEncoder encoder){
		List<Feature> features = Collections.emptyList();

		List<String> names = getFeatureNamesIn();
		if(names != null){
			features = new ArrayList<>();

			for(String featureNameIn : names){
				Feature feature = InitializerUtil.createWildcardFeature(featureNameIn, encoder);

				features.add(feature);
			}
		}

		return encodeFeatures(features, encoder);
	}

	@Override
	public List<Feature> encodeFeatures(List<Feature> features, SkLearnEncoder encoder){
		List<Object[]> fittedTransformers = getFittedTransformers();

		List<Feature> result = new ArrayList<>();

		for(Object[] fittedTransformer : fittedTransformers){
			Transformer transformer = getTransformer(fittedTransformer);

			List<Feature> rowFeatures = getFeatures(fittedTransformer, features, encoder);

			rowFeatures = transformer.encode(rowFeatures, encoder);

			result.addAll(rowFeatures);
		}

		return result;
	}

	@Override
	public PMML encodePMML(){
		return TransformerUtil.encodePMML(this);
	}

	public List<Object[]> getFittedTransformers(){
		return getTupleList("transformers_");
	}

	@Override
	public Boolean getSparseOutput(){
		return getBoolean("sparse_output_");
	}

	static
	protected String getName(Object[] fittedTransformer){
		return TupleUtil.extractStringElement(fittedTransformer, 0);
	}

	static
	protected Transformer getTransformer(Object[] fittedTransformer){
		CastFunction<Transformer> castFunction = new TransformerCastFunction<Transformer>(Transformer.class){

			@Override
			public Transformer apply(Object object){

				if(Objects.equals(SkLearnSteps.DROP, object)){
					return Drop.INSTANCE;
				} else

				if(Objects.equals(SkLearnSteps.PASSTHROUGH, object)){
					return PassThrough.INSTANCE;
				}

				return super.apply(object);
			}
		};

		return TupleUtil.extractElement(fittedTransformer, 1, castFunction);
	}

	static
	protected void setTransformer(Object[] fittedTransformer, Transformer transformer){
		fittedTransformer[1] = transformer;
	}

	static
	protected List<Feature> getFeatures(Object[] fittedTransformer, List<Feature> features, SkLearnEncoder encoder){
		Object columns = TupleUtil.extractElement(fittedTransformer, 2);

		// SkLearn 1.5+
		if(columns instanceof RemainderColsList){
			RemainderColsList remainderColsList = (RemainderColsList)columns;

			columns = remainderColsList.getData();
		} // End if

		if((columns instanceof String) || (columns instanceof Integer)){
			columns = Collections.singletonList(columns);
		} else

		if(columns instanceof HasArray){
			HasArray hasArray = (HasArray)columns;

			List<Object> values = new ArrayList<>();
			values.addAll(hasArray.getArrayContent());

			// Convert dense boolean array to sparse integer array (the indices of true values)
			for(int i = values.size() - 1; i > -1; i--){
				Object value = values.get(i);

				if(value instanceof Boolean){
					Boolean booleanValue = (Boolean)value;

					if(booleanValue){
						values.set(i, i);
					} else

					{
						values.remove(i);
					}
				}
			}

			columns = values;
		}

		return InitializerUtil.selectFeatures((List)columns, features, encoder);
	}
}