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
package org.jpmml.sklearn.testing;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Predicate;

import com.google.common.base.Equivalence;
import org.jpmml.converter.FieldNameUtil;
import org.jpmml.converter.testing.Datasets;
import org.jpmml.converter.testing.OptionsUtil;
import org.jpmml.evaluator.ResultField;
import org.jpmml.evaluator.testing.PMMLEquivalence;
import org.jpmml.model.visitors.VisitorBattery;
import org.junit.Test;
import sklearn.Estimator;
import sklearn.OutlierDetector;
import sklearn.tree.HasTreeOptions;

public class OutlierDetectorTest extends ValidatingSkLearnEncoderBatchTest implements SkLearnAlgorithms, Datasets {

	@Override
	public ValidatingSkLearnEncoderBatch createBatch(String algorithm, String dataset, Predicate<ResultField> columnFilter, Equivalence<Object> equivalence){
		ValidatingSkLearnEncoderBatch result = new ValidatingSkLearnEncoderBatch(algorithm, dataset, columnFilter, equivalence){

			@Override
			public OutlierDetectorTest getArchiveBatchTest(){
				return OutlierDetectorTest.this;
			}

			@Override
			public List<Map<String, Object>> getOptionsMatrix(){
				String algorithm = getAlgorithm();

				if((ISOLATION_FOREST).equals(algorithm)){
					Map<String, Object> options = new LinkedHashMap<>();
					options.put(HasTreeOptions.OPTION_PRUNE, new Boolean[]{false, true});

					return OptionsUtil.generateOptionsMatrix(options);
				}

				return super.getOptionsMatrix();
			}

			@Override
			public VisitorBattery getValidators(){
				VisitorBattery visitorBattery = super.getValidators();

				visitorBattery.add(ModelStatsInspector.class);

				return visitorBattery;
			}
		};

		return result;
	}

	@Test
	public void evaluateIsolationForestHousing() throws Exception {
		evaluate(ISOLATION_FOREST, HOUSING, excludeFields("rawAnomalyScore", "normalizedAnomalyScore", OutlierDetectorTest.predictedValue), new PMMLEquivalence(5e-12, 5e-12));
	}

	@Test
	public void evaluateOneClassSVMHousing() throws Exception {
		evaluate(ONE_CLASS_SVM, HOUSING, excludeFields(OutlierDetectorTest.predictedValue));
	}

	@Test
	public void evaluateSGDOneClassSVMIris() throws Exception {
		evaluate(SGD_ONE_CLASS_SVM, IRIS, excludeFields(OutlierDetectorTest.predictedValue));
	}

	private static final String predictedValue = FieldNameUtil.create(Estimator.FIELD_PREDICT, OutlierDetector.FIELD_OUTLIER);
}