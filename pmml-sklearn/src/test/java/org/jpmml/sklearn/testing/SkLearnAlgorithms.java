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
package org.jpmml.sklearn.testing;

public interface SkLearnAlgorithms {

	String ADA_BOOST = "AdaBoost";
	String ARD = "ARD";
	String BAYESIAN_ARD = "Bayesian" + ARD;
	String CQR = "CQR";
	String DECISION_TREE = "DecisionTree";
	String TRANSFORMED_DECISION_TREE = "Transformed" + DECISION_TREE;
	String DECISION_TREE_ENSEMBLE = DECISION_TREE + "Ensemble";
	String DUMMY = "Dummy";
	String ELASTIC_NET = "ElasticNet";
	String EXTRA_TREES = "ExtraTrees";
	String ESTIMATOR_CHAIN = "EstimatorChain";
	String MULTI_ESTIMATOR_CHAIN = "Multi" + ESTIMATOR_CHAIN;
	String GAMMA_REGRESSION = "GammaRegression";
	String GBDT_LM = "GBDTLM";
	String GBDT_LR = "GBDTLR";
	String GRADIENT_BOOSTING = "GradientBoosting";
	String HIST_GRADIENT_BOOSTING = "HistGradientBoosting";
	String HUBER = "Huber";
	String ISOLATION_FOREST = "IsolationForest";
	String ISOTONIC_REGRESSON = "IsotonicRegression";
	String ISOTONIC_REGRESSION_DECR = ISOTONIC_REGRESSON + "Decr";
	String ISOTONIC_REGRESSION_INCR = ISOTONIC_REGRESSON + "Incr";
	String K_MEANS = "KMeans";
	String MINIBATCH_K_MEANS = "MiniBatch" + K_MEANS;
	String KNN = "KNN";
	String MULTI_KNN = "Multi" + KNN;
	String LARS = "Lars";
	String LASSO = "Lasso";
	String LASSO_LARS = LASSO + LARS;
	String LINEAR_DISCRIMINANT_ANALYSIS = "LinearDiscriminantAnalysis";
	String LINEAR_REGRESSION = "LinearRegression";
	String MULTI_LINEAR_REGRESSION = "Multi" + LINEAR_REGRESSION;
	String TRANSFORMED_LINEAR_REGRESSION = "Transformed" + LINEAR_REGRESSION;
	String LINEAR_REGRESSION_CHAIN = LINEAR_REGRESSION + "Chain";
	String LINEAR_REGRESSION_ENSEMBLE = LINEAR_REGRESSION + "Ensemble";
	String LINEAR_SVC = "LinearSVC";
	String LINEAR_SVR = "LinearSVR";
	String MULTI_LINEAR_SVR = "Multi" + LINEAR_SVR;
	String LOGISTIC_REGRESSION = "LogisticRegression";
	String MULTI_LOGISTIC_REGRESSION = "Multi" + LOGISTIC_REGRESSION;
	String MULTINOMIAL_LOGISTIC_REGRESSION = "Multinomial" + LOGISTIC_REGRESSION;
	String OVR_LOGISTIC_REGRESSION = "OvR" + LOGISTIC_REGRESSION;
	String LOGISTIC_REGRESSION_CHAIN = LOGISTIC_REGRESSION + "Chain";
	String LOGISTIC_REGRESSION_ENSEMBLE = LOGISTIC_REGRESSION + "Ensemble";
	String MLP = "MLP";
	String MULTI_MLP = "Multi" + MLP;
	String NAIVE_BAYES = "NaiveBayes";
	String NU_SVC = "NuSVC";
	String NU_SVR = "NuSVR";
	String OMP = "OMP";
	String ONE_CLASS_SVM = "OneClassSVM";
	String ONE_VS_REST = "OneVsRest";
	String POISSON_REGRESSION = "PoissonRegression";
	String RANDOM_FOREST = "RandomForest";
	String RIDGE = "Ridge";
	String BAYESIAN_RIDGE = "Bayesian" + RIDGE;
	String RIDGE_ENSEMBLE = RIDGE + "Ensemble";
	String RULE_SET = "RuleSet";
	String SELECT_FIRST = "SelectFirst";
	String SGD = "SGD";
	String SGD_ONE_CLASS_SVM = "SGD" + ONE_CLASS_SVM;
	String SGD_LOG = SGD + "Log";
	String STACKING_ENSEMBLE = "StackingEnsemble";
	String SVC = "SVC";
	String SVR = "SVR";
	String THEIL_SEN = "TheilSen";
	String VOTING = "Voting";
	String VOTING_ENSEMBLE = VOTING + "Ensemble";
}