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

import org.jpmml.converter.Feature;
import org.jpmml.converter.ValueUtil;
import org.jpmml.sklearn.ClassDictUtil;
import org.jpmml.sklearn.FeatureMapper;
import sklearn.HasNumberOfFeatures;
import sklearn.Transformer;

public class SelectKBest extends Transformer implements HasNumberOfFeatures {

	public SelectKBest(String module, String name){
		super(module, name);
	}

	@Override
	public int getNumberOfFeatures(){
		List<? extends Number> scores = getScores();

		return scores.size();
	}

	@Override
	public List<Feature> encodeFeatures(List<String> ids, List<Feature> inputFeatures, FeatureMapper featureMapper){
		Object k = getK();
		List<? extends Number> scores = getScores();

		if(inputFeatures.size() != scores.size()){
			throw new IllegalArgumentException();
		} // End if

		if(("all").equals(k)){
			return inputFeatures;
		}

		List<FeatureScore> featureScores = new ArrayList<>();

		for(int i = 0; i < inputFeatures.size(); i++){
			Feature inputFeature = inputFeatures.get(i);
			Number score = scores.get(i);

			FeatureScore featureScore = new FeatureScore(inputFeature, score.doubleValue());

			featureScores.add(featureScore);
		}

		Collections.sort(featureScores, Collections.reverseOrder());

		List<Feature> features = new ArrayList<>();

		for(int i = 0, max = ValueUtil.asInt((Number)k); i < max; i++){
			FeatureScore featureScore = featureScores.get(i);

			Feature feature = featureScore.getFeature();

			features.add(feature);
		}

		return features;
	}

	public Object getK(){
		return get("k");
	}

	public List<? extends Number> getScores(){
		return (List)ClassDictUtil.getArray(this, "scores_");
	}

	static
	private class FeatureScore implements Comparable<FeatureScore> {

		private Feature feature;

		private double score;


		public FeatureScore(Feature feature, double score){
			setFeature(feature);
			setScore(score);
		}

		@Override
		public int compareTo(FeatureScore that){
			return Double.compare(this.getScore(), that.getScore());
		}

		public Feature getFeature(){
			return this.feature;
		}

		private void setFeature(Feature feature){
			this.feature = feature;
		}

		public double getScore(){
			return this.score;
		}

		private void setScore(double score){
			this.score = score;
		}
	}
}