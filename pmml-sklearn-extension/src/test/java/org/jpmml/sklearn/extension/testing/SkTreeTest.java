/*
 * Copyright (c) 2024 Villu Ruusmann
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

import org.jpmml.converter.FieldNameUtil;
import org.jpmml.converter.testing.Datasets;
import org.jpmml.evaluator.testing.PMMLEquivalence;
import org.jpmml.sklearn.testing.SkLearnEncoderBatchTest;
import org.junit.jupiter.api.Test;
import sklearn.Estimator;
import sklearn.OutlierDetector;

public class SkTreeTest extends SkLearnEncoderBatchTest implements Datasets {

	@Test
	public void evaluateObliqueDecisionTreeAudit() throws Exception {
		evaluate("ObliqueDecisionTree", AUDIT);
	}

	@Test
	public void evaluateObliqueRandomForestAudit() throws Exception {
		evaluate("ObliqueRandomForest", AUDIT);
	}

	@Test
	public void evaluateObliqueDecisionTreeAuto() throws Exception {
		evaluate("ObliqueDecisionTree", AUTO);
	}

	@Test
	public void evaluateObliqueRandomForestAuto() throws Exception {
		evaluate("ObliqueRandomForest", AUTO);
	}

	@Test
	public void evaluateExtendedIsolationForestHousing() throws Exception {
		evaluate("ExtendedIsolationForest", HOUSING, excludeFields("rawAnomalyScore", "normalizedAnomalyScore", SkTreeTest.predictedValue), new PMMLEquivalence(5e-12, 5e-12));
	}

	@Test
	public void evaluateObliqueDecisionTreeIris() throws Exception {
		evaluate("ObliqueDecisionTree", IRIS);
	}

	@Test
	public void evaluateObliqueRandomForestIris() throws Exception {
		evaluate("ObliqueRandomForest", IRIS);
	}

	private static final String predictedValue = FieldNameUtil.create(Estimator.FIELD_PREDICT, OutlierDetector.FIELD_OUTLIER);
}