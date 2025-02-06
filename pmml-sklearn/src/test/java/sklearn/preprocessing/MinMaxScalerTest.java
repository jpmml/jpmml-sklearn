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
package sklearn.preprocessing;

import org.dmg.pmml.PMMLFunctions;
import org.junit.jupiter.api.Test;

public class MinMaxScalerTest extends ScalerTest {

	@Test
	public void encode(){
		MinMaxScaler scaler = new MinMaxScaler("sklearn.preprocessing.data", "MinMaxScaler");
		scaler.put("min_", 2d);
		scaler.put("scale_", 6d);

		assertTransformedFeature(scaler, PMMLFunctions.ADD);

		scaler.put("min_", 0d);

		assertTransformedFeature(scaler, PMMLFunctions.MULTIPLY);

		scaler.put("scale_", 1d);

		assertSameFeature(scaler);
	}
}