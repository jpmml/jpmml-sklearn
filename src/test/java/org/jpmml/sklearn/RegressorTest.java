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
package org.jpmml.sklearn;

import org.junit.Test;

public class RegressorTest extends EstimatorTest {

	@Test
	public void evaluateDecisionTreeAuto() throws Exception {
		evaluate("DecisionTree", "Auto");
	}

	@Test
	public void evaluateDecisionTreeEnsembleAuto() throws Exception {
		evaluate("DecisionTreeEnsemble", "Auto");
	}

	@Test
	public void evaluateElasticNetAuto() throws Exception {
		evaluate("ElasticNet", "Auto");
	}

	@Test
	public void evaluateExtraTreesAuto() throws Exception {
		evaluate("ExtraTrees", "Auto");
	}

	@Test
	public void evaluateGradientBoostingAuto() throws Exception {
		evaluate("GradientBoosting", "Auto");
	}

	@Test
	public void evaluateLassoAuto() throws Exception {
		evaluate("Lasso", "Auto");
	}

	@Test
	public void evaluateLinearRegressionAuto() throws Exception {
		evaluate("LinearRegression", "Auto");
	}

	@Test
	public void evaluateLinearRegressionEnsembleAuto() throws Exception {
		evaluate("LinearRegressionEnsemble", "Auto");
	}

	@Test
	public void evaluateRandomForestAuto() throws Exception {
		evaluate("RandomForest", "Auto");
	}

	@Test
	public void evaluateRidgeAuto() throws Exception {
		evaluate("Ridge", "Auto");
	}

	@Test
	public void evaluateSGDHousing() throws Exception {
		evaluate("SGD", "Housing");
	}

	@Test
	public void evaluateSVRHousing() throws Exception {
		evaluate("SVR", "Housing");
	}

	@Test
	public void evaluateNuSVRHousing() throws Exception {
		evaluate("NuSVR", "Housing");
	}
}