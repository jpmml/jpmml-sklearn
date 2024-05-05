/*
 * Copyright (c) 2016 Villu Ruusmann
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
package xgboost.sklearn;

import java.io.ByteArrayInputStream;
import java.io.InputStream;
import java.nio.ByteOrder;

import org.dmg.pmml.PMML;
import org.jpmml.python.PythonObject;
import org.jpmml.sklearn.Encodable;
import org.jpmml.sklearn.SkLearnException;
import org.jpmml.xgboost.Learner;
import org.jpmml.xgboost.XGBoostUtil;
import pandas.core.DataFrame;

public class Booster extends PythonObject implements Encodable {

	private Learner learner = null;


	public Booster(String module, String name){
		super(module, name);
	}

	@Override
	public PMML encodePMML(){
		return BoosterUtil.encodePMML(this);
	}

	public Learner getLearner(ByteOrder byteOrder, String charset){

		if(this.learner == null){
			this.learner = loadLearner(byteOrder, charset);
		}

		return this.learner;
	}

	private Learner loadLearner(ByteOrder byteOrder, String charset){
		byte[] handle = getHandle();

		try(InputStream is = new ByteArrayInputStream(handle)){
			return XGBoostUtil.loadLearner(is, byteOrder, charset, "$.Model");
		} catch(Exception e){
			throw new SkLearnException("Failed to load XGBoost booster object", e);
		}
	}

	public Integer getBestNTreeLimit(){
		return getOptionalInteger("best_ntree_limit");
	}

	public DataFrame getFMap(){
		return getOptional("fmap", DataFrame.class);
	}

	public byte[] getHandle(){
		return get("handle", byte[].class);
	}
}
