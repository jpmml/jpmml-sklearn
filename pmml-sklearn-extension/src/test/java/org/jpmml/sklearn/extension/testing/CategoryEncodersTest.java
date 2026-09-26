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

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.function.Predicate;

import com.google.common.base.Equivalence;
import org.jpmml.converter.testing.Datasets;
import org.jpmml.converter.testing.Fields;
import org.jpmml.converter.testing.OptionsUtil;
import org.jpmml.evaluator.ResultField;
import org.jpmml.sklearn.testing.SkLearnEncoderBatch;
import org.jpmml.sklearn.testing.SkLearnEncoderBatchTest;
import org.junit.jupiter.api.Test;
import sklearn.tree.HasTreeOptions;

public class CategoryEncodersTest extends SkLearnEncoderBatchTest implements Datasets, Fields {

	@Override
	public SkLearnEncoderBatch createBatch(String algorithm, String dataset, Predicate<ResultField> columnFilter, Equivalence<Object> equivalence){
		SkLearnEncoderBatch result = new SkLearnEncoderBatch(algorithm, dataset, columnFilter, equivalence){

			@Override
			public CategoryEncodersTest getArchiveBatchTest(){
				return CategoryEncodersTest.this;
			}

			@Override
			public List<Map<String, Object>> getOptionsMatrix(){
				String algorithm = getAlgorithm();
				String dataset = getDataset();

				if(Objects.equals(AUDIT, dataset)){

					if(RANDOM_FOREST_ENCODERS.contains(algorithm)){
						Map<String, Object> options = new LinkedHashMap<>();
						options.put(HasTreeOptions.OPTION_COMPACT, new Boolean[]{false, true});
						options.put(HasTreeOptions.OPTION_NUMERIC, new Boolean[]{false, true});

						if(Objects.equals("CatBoostEncoder", algorithm)){
							options.put(HasTreeOptions.OPTION_PRUNE, new Boolean[]{false, true});
						}

						return OptionsUtil.generateOptionsMatrix(options);
					}
				} else

				if(Objects.equals(AUDIT_NA, dataset)){

					if(RANDOM_FOREST_ENCODERS.contains(algorithm)){
						Map<String, Object> options = new LinkedHashMap<>();
						options.put(HasTreeOptions.OPTION_COMPACT, new Boolean[]{false, true});
						options.put(HasTreeOptions.OPTION_ALLOW_MISSING, Boolean.TRUE);

						return OptionsUtil.generateOptionsMatrix(options);
					}
				}

				return super.getOptionsMatrix();
			}
		};

		return result;
	}

	@Test
	public void evaluateBalancedDecisionTreeEnsembleAudit() throws Exception {
		evaluate("BalancedDecisionTreeEnsemble", AUDIT);
	}

	@Test
	public void evaluateBalancedRandomForestAudit() throws Exception {
		evaluate("BalancedRandomForest", AUDIT);
	}

	@Test
	public void evaluateBase2EncoderAudit() throws Exception {
		evaluate("Base2Encoder", AUDIT);
	}

	@Test
	public void evaluateBase2EncoderAuditNA() throws Exception {
		evaluate("Base2Encoder", AUDIT_NA);
	}

	@Test
	public void evaluateBase3EncoderAudit() throws Exception {
		evaluate("Base3Encoder", AUDIT);
	}

	@Test
	public void evaluateBase3EncoderAuditNA() throws Exception {
		evaluate("Base3Encoder", AUDIT_NA);
	}

	@Test
	public void evaluateBase4EncoderAudit() throws Exception {
		evaluate("Base4Encoder", AUDIT);
	}

	@Test
	public void evaluateBase4EncoderAuditNA() throws Exception {
		evaluate("Base4Encoder", AUDIT_NA);
	}

	@Test
	public void evaluateBinaryEncoderAudit() throws Exception {
		evaluate("BinaryEncoder", AUDIT);
	}

	@Test
	public void evaluateBinaryEncoderAuditNA() throws Exception {
		evaluate("BinaryEncoder", AUDIT_NA);
	}

	@Test
	public void evaluateCatBoostEncoderAudit() throws Exception {
		evaluate("CatBoostEncoder", AUDIT);
	}

	@Test
	public void evaluateCatBoostEncoderAuditNA() throws Exception {
		evaluate("CatBoostEncoder", AUDIT_NA);
	}

	@Test
	public void evaluateCountEncoderAudit() throws Exception {
		evaluate("CountEncoder", AUDIT);
	}

 	@Test
	public void evaluateCountEncoderAuditNA() throws Exception {
		evaluate("CountEncoder", AUDIT_NA);
	}

	@Test
	public void evaluateLeaveOneOutEncoderAudit() throws Exception {
		evaluate("LeaveOneOutEncoder", AUDIT);
	}

	@Test
	public void evaluateLeaveOneOutEncoderAuditNA() throws Exception {
		evaluate("LeaveOneOutEncoder", AUDIT_NA);
	}

	@Test
	public void evaluateOneHotEncoderAudit() throws Exception {
		evaluate("OneHotEncoder", AUDIT);
	}

	@Test
	public void evaluateOneHotEncoderAuditNA() throws Exception {
		evaluate("OneHotEncoder", AUDIT_NA);
	}

	@Test
	public void evaluateOrdinalEncoderAudit() throws Exception {
		evaluate("OrdinalEncoder", AUDIT);
	}

	@Test
	public void evaluateOrdinalEncoderAuditNA() throws Exception {
		evaluate("OrdinalEncoder", AUDIT_NA);
	}

	@Test
	public void evaluateTargetEncoderAudit() throws Exception {
		evaluate("TargetEncoder", AUDIT);
	}

	@Test
	public void evaluateTargetEncoderAuditNA() throws Exception {
		evaluate("TargetEncoder", AUDIT_NA);
	}

	@Test
	public void evaluateWOEEncoderAudit() throws Exception {
		evaluate("WOEEncoder", AUDIT);
	}

	@Test
	public void evaluateWOEEncoderAuditNA() throws Exception {
		evaluate("WOEEncoder", AUDIT_NA);
	}

	private static final Set<String> RANDOM_FOREST_ENCODERS = Set.of("Base4Encoder", "BinaryEncoder", "CatBoostEncoder", "CountEncoder", "LeaveOneOutEncoder", "TargetEncoder", "WOEEncoder");
}