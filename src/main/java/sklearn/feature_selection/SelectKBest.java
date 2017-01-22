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
package sklearn.feature_selection;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import com.google.common.primitives.Booleans;
import org.jpmml.converter.ValueUtil;
import org.jpmml.sklearn.ClassDictUtil;
import sklearn.Selector;

public class SelectKBest extends Selector {

	public SelectKBest(String module, String name){
		super(module, name);
	}

	@Override
	public int getNumberOfFeatures(){
		int[] shape = getScoresShape();

		return shape[0];
	}

	@Override
	public List<Boolean> getSupportMask(){
		Object k = getK();
		List<? extends Number> scores = getScores();

		if(("all").equals(k)){
			return null;
		}

		List<Entry<Integer>> entries = new ArrayList<>();

		for(int i = 0; i < scores.size(); i++){
			Number score = scores.get(i);

			Entry<Integer> entry = new Entry<>(i, score.doubleValue());

			entries.add(entry);
		}

		Collections.sort(entries, Collections.reverseOrder());

		boolean[] result = new boolean[scores.size()];

		for(int i = 0, max = ValueUtil.asInt((Number)k); i < max; i++){
			Entry<Integer> entry = entries.get(i);

			result[entry.getId()] = true;
		}

		return Booleans.asList(result);
	}

	public Object getK(){
		return get("k");
	}

	public List<? extends Number> getScores(){
		return (List)ClassDictUtil.getArray(this, "scores_");
	}

	private int[] getScoresShape(){
		return ClassDictUtil.getShape(this, "scores_", 1);
	}

	static
	private class Entry<E> implements Comparable<Entry<E>> {

		private E id;

		private double score;


		public Entry(E id, double score){
			setId(id);
			setScore(score);
		}

		@Override
		public int compareTo(Entry<E> that){
			return Double.compare(this.getScore(), that.getScore());
		}

		public E getId(){
			return this.id;
		}

		private void setId(E id){
			this.id = id;
		}

		public double getScore(){
			return this.score;
		}

		private void setScore(double score){
			this.score = score;
		}
	}
}