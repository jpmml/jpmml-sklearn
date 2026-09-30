/*
 * Copyright (c) 2025 Villu Ruusmann
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
package sklearn.linear_model.glm;

import java.util.Arrays;

import org.dmg.pmml.regression.RegressionModel;
import sklearn.linear_model.LinearRegressor;

public class TweedieRegressor extends LinearRegressor {

	public TweedieRegressor(String module, String name){
		super(module, name);
	}

	@Override
	protected RegressionModel.NormalizationMethod getNormalizationMethod(){
		String link = getLink();

		switch(link){
			case TweedieRegressor.LINK_AUTO:
				{
					Number power = getPower();

					if(power.doubleValue() > 0d){
						return RegressionModel.NormalizationMethod.EXP;
					}

					return super.getNormalizationMethod();
				}
			case TweedieRegressor.LINK_IDENTITY:
				{
					return super.getNormalizationMethod();
				}
			case TweedieRegressor.LINK_LOG:
				{
					return RegressionModel.NormalizationMethod.EXP;
				}
			default:
				throw new IllegalArgumentException(link);
		}
	}

	public String getLink(){
		return getEnum("link", this::getString, Arrays.asList(TweedieRegressor.LINK_AUTO, TweedieRegressor.LINK_IDENTITY, TweedieRegressor.LINK_LOG));
	}

	public Number getPower(){
		return getNumber("power");
	}

	private static final String LINK_AUTO = "auto";
	private static final String LINK_IDENTITY = "identity";
	private static final String LINK_LOG = "log";
}