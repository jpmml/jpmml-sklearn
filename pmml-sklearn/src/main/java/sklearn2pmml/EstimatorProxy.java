/*
 * Copyright (c) 2017 Villu Ruusmann
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
package sklearn2pmml;

import java.util.List;
import java.util.Map;

import org.dmg.pmml.DataType;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.Model;
import org.dmg.pmml.OpType;
import org.jpmml.converter.Feature;
import org.jpmml.converter.Label;
import org.jpmml.converter.Schema;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn.Estimator;
import sklearn.EstimatorUtil;
import sklearn.HasClasses;
import sklearn.HasEstimator;
import sklearn.HasFeatureNamesIn;
import sklearn.Proxy;

public class EstimatorProxy extends Estimator implements HasClasses, HasEstimator<Estimator>, HasFeatureNamesIn, Proxy {

	public EstimatorProxy(){
		this("sklearn2pmml", "EstimatorProxy");
	}

	public EstimatorProxy(String module, String name){
		super(module, name);
	}

	@Override
	public MiningFunction getMiningFunction(){
		Estimator estimator = getEstimator();

		return estimator.getMiningFunction();
	}

	@Override
	public boolean isSupervised(){
		Estimator estimator = getEstimator();

		return estimator.isSupervised();
	}

	@Override
	public OpType getOpType(){
		Estimator estimator = getEstimator();

		return estimator.getOpType();
	}

	@Override
	public DataType getDataType(){
		Estimator estimator = getEstimator();

		return estimator.getDataType();
	}

	@Override
	public List<String> getFeatureNamesIn(){
		Estimator estimator = getEstimator();

		return estimator.getFeatureNamesIn();
	}

	@Override
	public int getNumberOfFeatures(){
		Estimator estimator = getEstimator();

		return estimator.getNumberOfFeatures();
	}

	@Override
	public int getNumberOfOutputs(){
		Estimator estimator = getEstimator();

		return estimator.getNumberOfOutputs();
	}

	@Override
	public List<?> getClasses(){
		Estimator estimator = getEstimator();

		return EstimatorUtil.getClasses(estimator);
	}

	@Override
	public boolean hasProbabilityDistribution(){
		Estimator estimator = getEstimator();

		return EstimatorUtil.hasProbabilityDistribution(estimator);
	}

	@Override
	public String getAlgorithmName(){
		Estimator estimator = getEstimator();

		return estimator.getAlgorithmName();
	}

	@Override
	public Label encodeLabel(List<String> names, SkLearnEncoder encoder){
		Estimator estimator = getEstimator();

		return estimator.encodeLabel(names, encoder);
	}

	@Override
	public Model encodeModel(Schema schema){
		Estimator estimator = getEstimator();

		return estimator.encodeModel(schema);
	}

	@Override
	public Schema configureSchema(Schema schema){
		Estimator estimator = getEstimator();

		return estimator.configureSchema(schema);
	}

	@Override
	public Model configureModel(Model model){
		Estimator estimator = getEstimator();

		return estimator.configureModel(model);
	}

	@Override
	public void checkLabel(Label label){
		Estimator estimator = getEstimator();

		estimator.checkLabel(label);
	}

	@Override
	public void checkFeatures(List<? extends Feature> features){
		Estimator estimator = getEstimator();

		estimator.checkFeatures(features);
	}

	@Override
	public Estimator getEstimator(){

		// SkLearn2PMML 0.54.0
		if(hasattr("estimator_")){
			return getEstimator("estimator_");
		} else

		// SkLearn2PMML 0.55.0+
		{
			return getEstimator("estimator");
		}
	}

	@Override
	public Map<String, ?> getPMMLOptions(){
		Estimator estimator = getEstimator();

		return estimator.getPMMLOptions();
	}

	@Override
	public EstimatorProxy setPMMLOptions(Map<String, ?> pmmlOptions){
		Estimator estimator = getEstimator();

		estimator.setPMMLOptions(pmmlOptions);

		return this;
	}

	@Override
	public Object getPMMLSegmentId(){
		Estimator estimator = getEstimator();

		return estimator.getPMMLSegmentId();
	}

	@Override
	public EstimatorProxy setPMMLSegmentId(Object pmmlSegmentId){
		Estimator estimator = getEstimator();

		estimator.setPMMLSegmentId(pmmlSegmentId);

		return this;
	}

	static
	public String formatExample(Estimator estimator){
		return
			"from sklearn2pmml import EstimatorProxy" + "\n" +
			"proxied_estimator = EstimatorProxy(" + estimator.getPythonName() + "(...))";
	}
}