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

import org.junit.Test;

public class StandardScalerTest extends ScalerTest {

	@Test
	public void encode(){
		StandardScaler scaler = new StandardScaler("sklearn.preprocessing.data", "StandardScaler");
		scaler.put("with_mean", Boolean.FALSE);
		scaler.put("with_std", Boolean.FALSE);
		scaler.put("mean_", 6d);
		scaler.put("std_", 2d);

		assertSameFeature(scaler);

		scaler.put("with_mean", Boolean.TRUE);
		scaler.put("with_std", Boolean.TRUE);

		assertTransformedFeature(scaler, "/");

		scaler.put("std_", 1d);

		assertTransformedFeature(scaler, "-");

		scaler.put("mean_", 0d);

		assertSameFeature(scaler);
	}
}