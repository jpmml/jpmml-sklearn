/*
 * Copyright (c) 2024 Villu Ruusmann
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
package sklearn.neural_network;

import java.util.Arrays;
import java.util.List;

interface MLPConstants {

	String ACTIVATION_IDENTITY = "identity";
	String ACTIVATION_LOGISTIC = "logistic";
	String ACTIVATION_RELU = "relu";
	String ACTIVATION_TANH = "tanh";

	List<String> ENUM_ACTIVATION = Arrays.asList(ACTIVATION_IDENTITY, ACTIVATION_LOGISTIC, ACTIVATION_RELU, ACTIVATION_TANH);
}