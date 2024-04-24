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
package sklearn2pmml.statsmodels;

import java.util.List;

import org.dmg.pmml.DataType;
import org.dmg.pmml.PMML;
import org.jpmml.converter.Feature;
import org.jpmml.converter.Label;
import org.jpmml.converter.Schema;
import org.jpmml.sklearn.SkLearnEncoder;
import org.jpmml.statsmodels.InterceptFeature;
import org.jpmml.statsmodels.StatsModelsEncoder;
import sklearn.Estimator;
import statsmodels.ResultsWrapper;

public class StatsModelsUtil {

	private StatsModelsUtil(){
	}

	static
	public <E extends Estimator & HasResults> PMML encodePMML(E estimator){
		StatsModelsEncoder encoder = new StatsModelsEncoder();

		ResultsWrapper resultsWrapper = estimator.getResults();

		return resultsWrapper.encodePMML(encoder);
	}

	static
	public Schema addConstant(Schema schema){
		SkLearnEncoder encoder = (SkLearnEncoder)schema.getEncoder();
		Label label = schema.getLabel();
		List<Feature> features = (List)schema.getFeatures();

		features.add(0, new InterceptFeature(encoder, "const", DataType.DOUBLE));

		return new Schema(encoder, label, features);
	}

	static
	public void initOnce(){

		if(!StatsModelsUtil.initialized){
			init();

			StatsModelsUtil.initialized = true;
		}
	}

	static
	private void init(){
		@SuppressWarnings("unused")
		StatsModelsEncoder encoder = new StatsModelsEncoder();
	}

	private static boolean initialized = false;
}