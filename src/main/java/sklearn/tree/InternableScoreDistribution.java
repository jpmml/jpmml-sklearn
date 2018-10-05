/*
 * Copyright (c) 2018 Villu Ruusmann
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
package sklearn.tree;

import java.util.Objects;

import org.dmg.pmml.ScoreDistribution;

public class InternableScoreDistribution extends ScoreDistribution {

	@Override
	public int hashCode(){
		int result = 0;

		result += (31 * result) + Objects.hashCode(this.getValue());
		result += (31 * result) + Double.hashCode(this.getRecordCount());
		result += (31 * result) + Objects.hashCode(this.getProbability());

		return result;
	}

	@Override
	public boolean equals(Object object){

		if(object instanceof InternableScoreDistribution){
			InternableScoreDistribution that = (InternableScoreDistribution)object;

			return Objects.equals(this.getValue(), that.getValue()) && (this.getRecordCount() == that.getRecordCount()) && Objects.equals(this.getProbability(), that.getProbability());
		}

		return false;
	}
}