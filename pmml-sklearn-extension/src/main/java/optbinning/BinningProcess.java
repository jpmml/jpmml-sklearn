/*
 * Copyright (c) 2022 Villu Ruusmann
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
package optbinning;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import com.google.common.collect.Iterables;
import com.google.common.collect.Maps;
import org.jpmml.converter.Feature;
import org.jpmml.python.CastFunction;
import org.jpmml.python.ClassDictUtil;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn.Initializer;
import sklearn.InitializerUtil;

public class BinningProcess extends Initializer {

	public BinningProcess(String module, String name){
		super(module, name);
	}

	@Override
	public List<Feature> initializeFeatures(SkLearnEncoder encoder){
		return encodeFeatures(Collections.emptyList(), encoder);
	}

	@Override
	public List<Feature> encodeFeatures(List<Feature> features, SkLearnEncoder encoder){
		List<Boolean> support = getSupport();

		Map<String, OptimalBinning> binnedVariables = getBinnedVariables();

		Map<String, Map<String, ?>> binningFitParams = getBinningFitParams();
		Map<String, Map<String, ?>> binningTransformParams = getBinningTransformParams();

		List<String> variableNames = getVariableNames();
		Map<String, String> variableDTypes = getVariableDTypes();
		Map<String, Map<String, ?>> variableStats = getVariableStats();

		ClassDictUtil.checkSize(support, variableNames, variableDTypes.values(), variableStats.values());

		List<Feature> result = new ArrayList<>();

		for(int i = 0; i < support.size(); i++){
			boolean flag = support.get(i);

			if(!flag){
				continue;
			}

			String variableName = variableNames.get(i);

			Feature feature = InitializerUtil.selectFeature(variableName, features, encoder);

			OptimalBinning optimalBinning = binnedVariables.get(variableName);

			Map<String, ?> fitParams = binningFitParams.get(variableName);
			Map<String, ?> transformParams = binningTransformParams.get(variableName);

			if(transformParams != null && !transformParams.isEmpty()){
				Collection<? extends Map.Entry<String, ?>> entries = transformParams.entrySet();

				for(Map.Entry<String, ?> entry : entries){
					String key = entry.getKey();
					Object value = entry.getValue();

					if(!optimalBinning.containsKey(key)){
						optimalBinning.put(key, value);
					}
				}
			}

			feature = Iterables.getOnlyElement(optimalBinning.encodeFeatures(Collections.singletonList(feature), encoder));

			result.add(feature);
		}

		return result;
	}

	public Map<String, OptimalBinning> getBinnedVariables(){
		Map<String, ?> binnedVariables = getDict("_binned_variables");

		CastFunction<OptimalBinning> castFunction = new CastFunction<OptimalBinning>(OptimalBinning.class){

			@Override
			protected String formatMessage(Object object){
				return "The binning object (" + ClassDictUtil.formatClass(object) + ") is not a supported";
			}
		};

		return Maps.transformValues(binnedVariables, castFunction);
	}

	public Map<String, Map<String, ?>> getBinningFitParams(){
		return getBinningParams("binning_fit_params");
	}

	public Map<String, Map<String, ?>> getBinningTransformParams(){
		return getBinningParams("binning_transform_params");
	}

	private Map<String, Map<String, ?>> getBinningParams(String name){
		Object binningParams = get(name);

		if(binningParams == null){
			return Collections.emptyMap();
		}

		return (Map)getDict(name);
	}

	public List<Boolean> getSupport(){
		return (List)getArray("_support", Boolean.class);
	}

	public List<String> getVariableNames(){
		return (List)getListLike("variable_names", String.class);
	}

	public Map<String, String> getVariableDTypes(){
		return (Map)getDict("_variable_dtypes");
	}

	public Map<String, Map<String, ?>> getVariableStats(){
		return (Map)getDict("_variable_stats");
	}
}