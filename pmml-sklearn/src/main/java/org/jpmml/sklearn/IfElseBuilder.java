/*
 * Copyright (c) 2023 Villu Ruusmann
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

import org.dmg.pmml.Apply;
import org.dmg.pmml.Expression;
import org.dmg.pmml.PMMLFunctions;
import org.jpmml.converter.PMMLUtil;

public class IfElseBuilder {

	private Apply apply = null;

	private Apply prevIfApply = null;

	private int size = 0;


	public IfElseBuilder(){
	}

	public IfElseBuilder add(Expression condition, Expression result){
		Apply ifApply = PMMLUtil.createApply(PMMLFunctions.IF, condition, result);

		if(this.apply == null){
			this.apply = ifApply;
		} // End if

		if(this.prevIfApply != null){
			this.prevIfApply.addExpressions(ifApply);
		}

		this.prevIfApply = ifApply;

		this.size += 1;

		return this;
	}

	public IfElseBuilder terminate(Expression result){

		if(this.prevIfApply == null){
			throw new IllegalStateException();
		}

		this.prevIfApply.addExpressions(result);

		return this;
	}

	public boolean isEmpty(){
		return (this.size == 0);
	}

	public Apply build(){

		if(this.apply == null){
			throw new IllegalStateException();
		}

		return this.apply;
	}
}