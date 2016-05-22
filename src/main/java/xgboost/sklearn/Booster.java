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

import net.razorvine.pickle.objects.ClassDict;
import org.jpmml.xgboost.Learner;
import org.jpmml.xgboost.XGBoostUtil;

public class Booster extends ClassDict {

	private Learner learner = null;


	public Booster(String module, String name){
		super(module, name);
	}

	public Learner getLearner(){

		if(this.learner == null){
			this.learner = loadLearner();
		}

		return this.learner;
	}

	private Learner loadLearner(){
		byte[] handle = getHandle();

		try(InputStream is = new ByteArrayInputStream(handle)){
			return XGBoostUtil.loadLearner(is);
		} catch(IOException ioe){
			throw new RuntimeException(ioe);
		}
	}

	public byte[] getHandle(){
		return (byte[])get("handle");
	}
}
