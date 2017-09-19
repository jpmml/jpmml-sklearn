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
package sklearn;

import org.dmg.pmml.DataType;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.Model;
import org.dmg.pmml.OpType;
import org.dmg.pmml.Visitor;
import org.jpmml.converter.Schema;
import org.jpmml.converter.ValueUtil;
import org.jpmml.sklearn.SkLearnEncoder;
import org.jpmml.sklearn.TreeModelProducer;
import org.jpmml.sklearn.visitors.TreeModelCompactor;

abstract
public class Estimator extends BaseEstimator implements HasNumberOfFeatures {

	public Estimator(String module, String name){
		super(module, name);
	}

	abstract
	public MiningFunction getMiningFunction();

	abstract
	public boolean isSupervised();

	abstract
	public Model encodeModel(Schema schema);

	public Model encodeModel(Schema schema, SkLearnEncoder encoder){
		Model model = encodeModel(schema);

		if(this instanceof TreeModelProducer){
			Boolean compact = (Boolean)getOption(TreeModelProducer.OPTION_COMPACT, Boolean.FALSE);

			if(compact){
				Visitor visitor = new TreeModelCompactor();

				visitor.applyTo(model);
			}
		}

		return model;
	}

	@Override
	public int getNumberOfFeatures(){
		return ValueUtil.asInt((Number)get("n_features_"));
	}

	public OpType getOpType(){
		return OpType.CONTINUOUS;
	}

	public DataType getDataType(){
		return DataType.DOUBLE;
	}
}