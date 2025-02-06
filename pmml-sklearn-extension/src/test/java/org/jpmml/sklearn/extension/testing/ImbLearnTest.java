/*
 * Copyright (c) 2020 Villu Ruusmann
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
package org.jpmml.sklearn.extension.testing;

import org.jpmml.converter.testing.Datasets;
import org.jpmml.sklearn.testing.SkLearnEncoderBatchTest;
import org.junit.jupiter.api.Test;

public class ImbLearnTest extends SkLearnEncoderBatchTest implements Datasets {

	@Test
	public void evaluateADASYNIris() throws Exception {
		evaluate("ADASYN", IRIS);
	}

	@Test
	public void evaluateClusterCentroidsIris() throws Exception {
		evaluate("ClusterCentroids", IRIS);
	}

	@Test
	public void evaluateNearMissIris() throws Exception {
		evaluate("NearMiss", IRIS);
	}

	@Test
	public void evaluateOneSidedSelectionIris() throws Exception {
		evaluate("OneSidedSelection", IRIS);
	}

	@Test
	public void evaluateRandomOverSamplerIris() throws Exception {
		evaluate("RandomOverSampler", IRIS);
	}

	@Test
	public void evaluateRandomUnderSamplerIris() throws Exception {
		evaluate("RandomUnderSampler", IRIS);
	}

	@Test
	public void evaluateSMOTEIris() throws Exception {
		evaluate("SMOTE", IRIS);
	}

	@Test
	public void evaluateSMOTEENNIris() throws Exception {
		evaluate("SMOTEENN", IRIS);
	}

	@Test
	public void evaluateSMOTETomekIris() throws Exception {
		evaluate("SMOTETomek", IRIS);
	}

	@Test
	public void evaluateTomekLinksIris() throws Exception {
		evaluate("TomekLinks", IRIS);
	}
}