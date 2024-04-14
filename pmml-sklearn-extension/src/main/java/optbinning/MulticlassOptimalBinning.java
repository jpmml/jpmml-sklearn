/*
 * Copyright (c) 2022 Villu Ruusmann
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
package optbinning;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;

import com.google.common.math.DoubleMath;
import org.jpmml.converter.CMatrixUtil;

public class MulticlassOptimalBinning extends OptimalBinning {

	public MulticlassOptimalBinning(String module, String name){
		super(module, name);
	}

	@Override
	public List<Double> getCategoriesOut(){
		String metric = getMetric();
		Integer numberOfClasses = getNumberOfClasses();
		List<Integer> numberOfEvents = getNumberOfEvents();

		switch(metric){
			case MulticlassOptimalBinning.METRIC_MEAN_WOE:
				break;
			default:
				throw new IllegalArgumentException(metric);
		}

		int cols = numberOfClasses;
		int rows = numberOfEvents.size() / numberOfClasses;

		List<List<Integer>> eventCountsByColumn = new ArrayList<>();

		for(int col = 0; col < cols; col++){
			List<Integer> eventCounts = CMatrixUtil.getColumn(numberOfEvents, rows, cols, col);

			eventCountsByColumn.add(eventCounts);
		}

		List<Integer> numberOfRecords = new ArrayList<>();

		for(int row = 0; row < rows; row++){
			List<Integer> eventCounts = CMatrixUtil.getRow(numberOfEvents, rows, cols, row);

			numberOfRecords.add(OptimalBinningUtil.sumExact(eventCounts));
		}

		List<List<Integer>> nonEventCountsByColumn = new ArrayList<>();

		for(int col = 0; col < cols; col++){
			List<Integer> nonEventCounts = new ArrayList<>();

			List<Integer> eventCounts = eventCountsByColumn.get(col);
			for(int row = 0; row < rows; row++){
				nonEventCounts.add(numberOfRecords.get(row) - eventCounts.get(row));
			}

			nonEventCountsByColumn.add(nonEventCounts);
		}

		List<List<Double>> woesByColumn = new ArrayList<>();

		for(int col = 0; col < cols; col++){
			List<Double> woes = new ArrayList<>();

			List<Integer> eventCounts = eventCountsByColumn.get(col);
			List<Integer> nonEventCounts = nonEventCountsByColumn.get(col);

			double constant = ((double)OptimalBinningUtil.sumExact(eventCounts)  / (double)OptimalBinningUtil.sumExact(nonEventCounts));

			for(int row = 0; row < rows; row++){
				double eventRate = (double)eventCounts.get(row) / (double)numberOfRecords.get(row);

				double woe = Math.log(((1d / eventRate) - 1d) * constant);

				if(Double.isNaN(woe)){
					woe = 0d;
				}

				woes.add(woe);
			}

			woesByColumn.add(woes);
		}

		List<Double> result = new ArrayList<>();

		for(int row = 0; row < rows; row++){
			List<Double> woesByRow = new ArrayList<>();

			for(int col = 0; col < cols; col++){
				List<Double> woes = woesByColumn.get(col);

				woesByRow.add(woes.get(row));
			}

			result.add(DoubleMath.mean(woesByRow));
		}

		return result;
	}

	@Override
	public String getDefaultMetric(){
		return MulticlassOptimalBinning.METRIC_MEAN_WOE;
	}

	@Override
	public Collection<String> getSupportedMetrics(){
		return Arrays.asList(MulticlassOptimalBinning.METRIC_MEAN_WOE);
	}

	public Integer getNumberOfClasses(){
		return getInteger("_n_classes");
	}

	@Override
	public List<Integer> getNumberOfEvents(){
		return getIntegerArray("_n_event");
	}

	private static final String METRIC_MEAN_WOE = "mean_woe";
}