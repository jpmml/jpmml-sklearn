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
package category_encoders;

import java.util.Arrays;
import java.util.List;

interface BaseEncoderConstants {

	String HANDLEMISSING_ERROR = "error";
	String HANDLEMISSING_RETURN_NAN = "return_nan";
	String HANDLEMISSING_VALUE = "value";

	List<String> ENUM_HANDLEMISSING = Arrays.asList(HANDLEMISSING_ERROR, HANDLEMISSING_RETURN_NAN, HANDLEMISSING_VALUE);

	String HANDLEUNKNOWN_ERROR = "error";
	String HANDLEUNKNOWN_VALUE = "value";

	List<String> ENUM_HANDLEUNKNOWN = Arrays.asList(HANDLEUNKNOWN_ERROR, HANDLEUNKNOWN_VALUE);
}