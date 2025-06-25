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
package sklearn2pmml;

public interface SkLearn2PMMLFields {

	String PMML_BASE_CLASS = "pmml_base_class_";
	String PMML_CLASSES = "pmml_classes_";
	String PMML_FEATURE_IMPORTANCES = "pmml_feature_importances_";
	String PMML_NAME = "pmml_name_";
	String PMML_CUSTOMIZATIONS = "pmml_customizations_";
	String PMML_OPTIONS = "pmml_options_";
	String PMML_SEGMENT_ID = "pmml_segment_id_";
}