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

public class RobustScalerTest extends ScalerTest {

	@Test
	public void encode(){
		RobustScaler scaler = new RobustScaler("sklearn.preprocessing.data", "RobustScaler");
		scaler.put("with_centering", Boolean.FALSE);
		scaler.put("with_scaling", Boolean.FALSE);
		scaler.put("center_", 6);
		scaler.put("scale_", 2);

		assertSameFeature(scaler);

		scaler.put("with_centering", Boolean.TRUE);
		scaler.put("with_scaling", Boolean.TRUE);

		assertTransformedFeature(scaler, "/");

		scaler.put("scale_", 1);

		assertTransformedFeature(scaler, "-");

		scaler.put("center_", 0);

		assertSameFeature(scaler);
	}
}