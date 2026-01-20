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
package sklearn2pmml;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import org.dmg.pmml.DataType;
import org.dmg.pmml.Extension;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.MiningSchema;
import org.dmg.pmml.Output;
import org.dmg.pmml.OutputField;
import org.dmg.pmml.PMMLObject;
import org.dmg.pmml.Targets;
import org.dmg.pmml.regression.RegressionModel;
import org.dmg.pmml.regression.RegressionTable;
import org.jpmml.converter.CategoricalLabel;
import org.jpmml.converter.ModelUtil;
import org.jpmml.model.ReflectionUtil;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class CustomizationTest {

	@Test
	public void customize() throws Exception {
		CategoricalLabel categoricalLabel = new CategoricalLabel(DataType.STRING, Arrays.asList("no", "yes"));

		MiningSchema miningSchema = ModelUtil.createMiningSchema(categoricalLabel);
		Output output = ModelUtil.createProbabilityOutput(DataType.DOUBLE, categoricalLabel);
		Targets targets = new Targets();

		RegressionModel regressionModel = new RegressionModel(MiningFunction.CLASSIFICATION, miningSchema, null)
			.setOutput(output)
			.setTargets(targets);

		assertFalse(regressionModel.hasExtensions());

		Extension insertExtension = new Extension("customized", "true");

		assertFalse(regressionModel.hasRegressionTables());

		RegressionModel updateRegressionModel = new RegressionModel()
			.addRegressionTables(new RegressionTable().setTargetCategory("no"))
			.addRegressionTables(new RegressionTable().setTargetCategory("yes"));

		List<OutputField> outputFields = output.getOutputFields();

		for(OutputField outputField : outputFields){
			assertTrue(outputField.isFinalResult());
		}

		List<String> probabilityNames = outputFields.stream()
			.map(OutputField::requireName)
			.collect(Collectors.toList());

		assertEquals(Arrays.asList("probability(no)", "probability(yes)"), probabilityNames);

		OutputField updateNoField = new OutputField()
			.setName("p(no)")
			.setFinalResult(false);

		OutputField updateYesField = new OutputField()
			.setName("p(yes)")
			.setFinalResult(false);

		assertNotNull(regressionModel.getTargets());

		List<Customization> customizations = Arrays.asList(
			Customization.createInsert(null, CustomizationUtil.formatPMML(insertExtension)),
			Customization.createUpdate(null, CustomizationUtil.formatPMML(updateRegressionModel)),
			Customization.createUpdate("//:Output/*[1]", CustomizationUtil.formatPMML(updateNoField)),
			Customization.createUpdate("//:OutputField[@name='probability(yes)']", CustomizationUtil.formatPMML(updateYesField)),
			Customization.createDelete("//:Targets")
		);

		CustomizationUtil.customize(regressionModel, customizations);

		List<Extension> extensions = regressionModel.getExtensions();

		checkList(Arrays.asList(insertExtension), extensions);

		List<RegressionTable> regressionTables = regressionModel.getRegressionTables();

		checkList(updateRegressionModel.getRegressionTables(), regressionTables);

		for(OutputField outputField : outputFields){
			assertFalse(outputField.isFinalResult());
		}

		probabilityNames = outputFields.stream()
			.map(OutputField::requireName)
			.collect(Collectors.toList());

		assertEquals(Arrays.asList("p(no)", "p(yes)"), probabilityNames);

		assertNull(regressionModel.getTargets());

		OutputField updateNoFieldStep1 = new OutputField()
			.setName("probability(NO)");

		OutputField updateNoFieldStep2 = new OutputField()
			.setFinalResult(true);

		OutputField updateYesFieldStep1 = new OutputField()
			.setName("probability(YES)");

		OutputField updateYesFieldStep2 = new OutputField()
			.setFinalResult(true);

		// Touch the same element multiple times during a customization session
		customizations = Arrays.asList(
			Customization.createUpdate("//:Output/*[1]", CustomizationUtil.formatPMML(updateNoFieldStep1)),
			Customization.createUpdate("//:Output/*[1]", CustomizationUtil.formatPMML(updateNoFieldStep2)),
			Customization.createUpdate("//:Output/*[2]", CustomizationUtil.formatPMML(updateYesFieldStep1)),
			Customization.createUpdate("//:Output/*[2]", CustomizationUtil.formatPMML(updateYesFieldStep2))
		);

		CustomizationUtil.customize(regressionModel, customizations);

		for(OutputField outputField : outputFields){
			assertTrue(outputField.isFinalResult());
		}

		probabilityNames = outputFields.stream()
			.map(OutputField::requireName)
			.collect(Collectors.toList());

		assertEquals(Arrays.asList("probability(NO)", "probability(YES)"), probabilityNames);
	}

	static
	private <E extends PMMLObject> void checkList(List<E> expected, List<E> actual){
		assertEquals(expected.size(), actual.size());

		for(int i = 0; i < expected.size(); i++){
			assertTrue(ReflectionUtil.equals(expected.get(i), actual.get(i)));
		}
	}
}