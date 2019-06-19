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
package sklearn2pmml.ensemble;

import java.util.List;

import com.google.common.collect.Iterables;
import org.dmg.pmml.DataType;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.Model;
import org.dmg.pmml.OpType;
import org.dmg.pmml.mining.MiningModel;
import org.dmg.pmml.regression.RegressionModel;
import org.jpmml.converter.CategoricalLabel;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.Schema;
import org.jpmml.converter.SchemaUtil;
import org.jpmml.converter.mining.MiningModelUtil;
import sklearn.Classifier;
import sklearn.linear_model.logistic.LogisticRegression;
import sklearn.preprocessing.MultiOneHotEncoder;

public class GBDTLRClassifier extends Classifier {

	public GBDTLRClassifier(String module, String name){
		super(module, name);
	}

	@Override
	public List<?> getClasses(){
		Classifier gbdt = getGBDT();

		return gbdt.getClasses();
	}

	@Override
	public Model encodeModel(Schema schema){
		Classifier gbdt = getGBDT();
		MultiOneHotEncoder ohe = getOHE();
		LogisticRegression lr = getLR();

		CategoricalLabel categoricalLabel = (CategoricalLabel)schema.getLabel();

		SchemaUtil.checkSize(2, categoricalLabel);

		List<? extends Number> coef = lr.getCoef();
		List<? extends Number> intercept = lr.getIntercept();

		Schema segmentSchema = schema.toAnonymousSchema();

		MiningModel miningModel = GBDTUtil.encodeModel(gbdt, ohe, coef, Iterables.getOnlyElement(intercept), segmentSchema)
			.setOutput(ModelUtil.createPredictedOutput(FieldName.create("decisionFunction"), OpType.CONTINUOUS, DataType.DOUBLE));

		return MiningModelUtil.createBinaryLogisticClassification(miningModel, 1d, 0d, RegressionModel.NormalizationMethod.LOGIT, true, schema);
	}

	public Classifier getGBDT(){
		return get("gbdt_", Classifier.class);
	}

	public LogisticRegression getLR(){
		return get("lr_", LogisticRegression.class);
	}

	public MultiOneHotEncoder getOHE(){
		return get("ohe_", MultiOneHotEncoder.class);
	}
}