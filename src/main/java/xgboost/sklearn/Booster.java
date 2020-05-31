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
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteOrder;

import org.jpmml.python.PythonObject;
import org.jpmml.xgboost.Learner;
import org.jpmml.xgboost.XGBoostUtil;

public class Booster extends PythonObject {

	private Learner learner = null;


	public Booster(String module, String name){
		super(module, name);
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
			return XGBoostUtil.loadLearner(is, byteOrder, charset);
		} catch(IOException ioe){
			throw new RuntimeException(ioe);
		}
	}

	public byte[] getHandle(){
		return get("handle", byte[].class);
	}
}
