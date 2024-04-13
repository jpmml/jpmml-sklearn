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
package sklearn2pmml;

import java.util.Arrays;

import org.jpmml.python.PythonObject;

public class Customization extends PythonObject {

	public Customization(){
		this("sklearn2pmml.customization", "Customization");
	}

	public Customization(String module, String name){
		super(module, name);
	}

	public String getCommand(){
		return getEnum("command", this::getString, Arrays.asList(Customization.COMMAND_INSERT, Customization.COMMAND_UPDATE, Customization.COMMAND_DELETE));
	}

	public Customization setCommand(String command){
		setattr("command", command);

		return this;
	}

	public String getXPathExpr(){
		return getOptionalString("xpath_expr");
	}

	public Customization setXPathExpr(String xPathExpr){
		setattr("xpath_expr", xPathExpr);

		return this;
	}

	public String getPMMLElement(){
		return getOptionalString("pmml_element");
	}

	public Customization setPMMLElement(String pmmlElement){
		setattr("pmml_element", pmmlElement);

		return this;
	}

	static
	public Customization createInsert(String xPathExpr, String pmmlElement){
		return new Customization()
			.setCommand(Customization.COMMAND_INSERT)
			.setXPathExpr(xPathExpr)
			.setPMMLElement(pmmlElement);
	}

	static
	public Customization createUpdate(String xPathExpr, String pmmlElement){
		return new Customization()
			.setCommand(Customization.COMMAND_UPDATE)
			.setXPathExpr(xPathExpr)
			.setPMMLElement(pmmlElement);
	}

	static
	public Customization createDelete(String xPathExpr){
		return new Customization()
			.setCommand(Customization.COMMAND_DELETE)
			.setXPathExpr(xPathExpr);
	}

	public static final String COMMAND_DELETE = "delete";
	public static final String COMMAND_INSERT = "insert";
	public static final String COMMAND_UPDATE = "update";
}