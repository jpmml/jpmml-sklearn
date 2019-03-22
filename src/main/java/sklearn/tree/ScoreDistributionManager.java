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

import java.util.ArrayList;
import java.util.List;

import com.google.common.collect.Interner;
import com.google.common.collect.Interners;
import org.dmg.pmml.ScoreDistribution;
import org.jpmml.converter.CategoricalLabel;

public class ScoreDistributionManager {

	private Interner<ScoreDistribution> interner = Interners.newStrongInterner();


	public List<ScoreDistribution> createScoreDistribution(CategoricalLabel categoricalLabel, double[] recordCounts){
		List<ScoreDistribution> result = new ArrayList<>();

		for(int i = 0; i < categoricalLabel.size(); i++){
			Object value = categoricalLabel.getValue(i);
			double recordCount = recordCounts[i];

			ScoreDistribution scoreDistribution = new InternableScoreDistribution()
				.setValue(value)
				.setRecordCount(recordCount);

			scoreDistribution = intern(scoreDistribution);

			result.add(scoreDistribution);
		}

		return result;
	}

	public ScoreDistribution intern(ScoreDistribution scoreDistribution){
		return this.interner.intern(scoreDistribution);
	}
}