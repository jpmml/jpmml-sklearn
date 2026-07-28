/*
 * Copyright (c) 2026 Villu Ruusmann
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
import java.util.Arrays;
import java.util.List;

import org.dmg.pmml.DataType;
import org.jpmml.converter.Feature;
import org.jpmml.sklearn.SkLearnEncoder;
import org.junit.jupiter.api.Test;
import sklearn.compose.ColumnTransformer;
import sklearn.pipeline.SkLearnPipeline;
import sklearn.preprocessing.OneHotEncoder;
import sklearn.preprocessing.StandardScaler;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;

public class TransformerTest extends StepTest {

	@Test
	public void encodeColumnTransformer(){
		List<Step> scalerParents = new ArrayList<>();
		List<Step> encoderParents = new ArrayList<>();

		Transformer scaler = new StandardScaler(null, null){

			@Override
			public int getNumberOfFeatures(){
				return 1;
			}

			@Override
			public List<Feature> encodeFeatures(List<Feature> features, SkLearnEncoder encoder){
				assertEquals(1, features.size());

				scalerParents.addAll(collectParents(this));

				return features;
			}
		};

		Transformer encoder = new OneHotEncoder(null, null){

			@Override
			public DataType getDataType(){
				return DataType.INTEGER;
			}

			@Override
			public List<Feature> encodeFeatures(List<Feature> features, SkLearnEncoder encoder){
				assertEquals(1, features.size());

				encoderParents.addAll(collectParents(this));

				return features;
			}
		};

		ColumnTransformer columnTransformer = new ColumnTransformer(null, null){

			@Override
			public List<Object[]> getFittedTransformers(){
				return Arrays.asList(
					new Object[]{"scaler", scaler, "x1"},
					new Object[]{"encoder", encoder, "x2"}
				);
			}
		};

		SkLearnPipeline pipeline = createPipeline("columnTransformer", columnTransformer);

		pipeline.encodePMML();

		assertEquals(2, scalerParents.size());

		assertSame(columnTransformer, scalerParents.get(0));
		assertSame(pipeline, scalerParents.get(1));

		assertEquals(2, encoderParents.size());

		assertSame(columnTransformer, encoderParents.get(0));
		assertSame(pipeline, encoderParents.get(1));
	}
}
