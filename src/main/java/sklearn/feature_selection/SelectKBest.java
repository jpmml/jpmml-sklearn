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
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import org.jpmml.converter.Feature;
import org.jpmml.converter.ValueUtil;
import org.jpmml.sklearn.ClassDictUtil;
import sklearn.Selector;

public class SelectKBest extends Selector {

	public SelectKBest(String module, String name){
		super(module, name);
	}

	@Override
	public int getNumberOfFeatures(){
		List<? extends Number> scores = getScores();

		return scores.size();
	}

	@Override
	public int[] selectFeatures(List<Feature> inputFeatures){
		Object k = getK();
		List<? extends Number> scores = getScores();

		if(inputFeatures.size() != scores.size()){
			throw new IllegalArgumentException();
		} // End if

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

		int[] result = new int[ValueUtil.asInt((Number)k)];

		for(int i = 0; i < result.length; i++){
			Entry<Integer> entry = entries.get(i);

			result[i] = entry.getId();
		}

		Arrays.sort(result);

		return result;
	}

	public Object getK(){
		return get("k");
	}

	public List<? extends Number> getScores(){
		return (List)ClassDictUtil.getArray(this, "scores_");
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