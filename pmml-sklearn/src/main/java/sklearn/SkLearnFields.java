/*
 * Copyright (c) 2021 Villu Ruusmann
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
package sklearn;

public interface SkLearnFields {

	String CLASSES = "classes_";
	String FEATURE_IMPORTANCES = "feature_importances_";
	String FEATURE_NAMES_IN = "feature_names_in_";
	String N_CLASSES = "n_classes_";
	String N_FEATURES = "n_features_";
	String N_FEATURES_IN = "n_features_in_";
	String N_OUTPUTS = "n_outputs_";
	String SKLEARN_TAGS = "_sklearn_tags";
	String SKLEARN_VERSION = "_sklearn_version";
}