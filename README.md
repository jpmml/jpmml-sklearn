JPMML-SkLearn
=============

Java library and command-line application for converting [Scikit-Learn](https://scikit-learn.org/) pipelines to PMML.

# Table of Contents #

* [Features](#features)
* [Prerequisites](#prerequisites)
  * [The Python side of operations](#the-python-side-of-operations)
  * [The JPMML-SkLearn side of operations](#the-jpmml-sklearn-side-of-operations)
* [Installation](#installation)
* [Usage](#usage)
  * [The Python side of operations](#the-python-side-of-operations-1)
  * [The JPMML-SkLearn side of operations](#the-jpmml-sklearn-side-of-operations-1)
* [Documentation](#documentation)
* [License](#license)
* [Additional information](#additional-information)

# Features #

* Supported Estimator and Transformer types:
  * Clustering:
    * [`cluster.KMeans`](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
    * [`cluster.MiniBatchKMeans`](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html)
  * Composite Estimators:
    * [`compose.ColumnTransformer`](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html)
    * [`compose.TransformedTargetRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.compose.TransformedTargetRegressor.html)
  * Matrix Decomposition:
    * [`decomposition.PCA`](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
    * [`decomposition.IncrementalPCA`](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.IncrementalPCA.html)
    * [`decomposition.TruncatedSVD`](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html)
  * Discriminant Analysis:
    * [`discriminant_analysis.LinearDiscriminantAnalysis`](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html)
  * Dummies:
    * [`dummy.DummyClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html)
    * [`dummy.DummyRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyRegressor.html)
  * Ensemble Methods:
    * [`ensemble.AdaBoostRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html)
    * [`ensemble.BaggingClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html)
    * [`ensemble.BaggingRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingRegressor.html)
    * [`ensemble.ExtraTreesClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html)
    * [`ensemble.ExtraTreesRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html)
    * [`ensemble.GradientBoostingClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)
    * [`ensemble.GradientBoostingRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html)
    * [`ensemble.HistGradientBoostingClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html)
    * [`ensemble.HistGradientBoostingRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html)
    * [`ensemble.IsolationForest`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)
    * [`ensemble.RandomForestClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
    * [`ensemble.RandomForestRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
    * [`ensemble.StackingClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html)
    * [`ensemble.StackingRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingRegressor.html)
    * [`ensemble.VotingClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html)
    * [`ensemble.VotingRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingRegressor.html)
  * Feature Extraction:
    * [`feature_extraction.DictVectorizer`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.DictVectorizer.html)
    * [`feature_extraction.text.CountVectorizer`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)
    * [`feature_extraction.text.TfidfVectorizer`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
  * Feature Selection:
    * [`feature_selection.GenericUnivariateSelect`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.GenericUnivariateSelect.html) (only via `sklearn2pmml.SelectorProxy`)
    * [`feature_selection.RFE`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html) (only via `sklearn2pmml.SelectorProxy`)
    * [`feature_selection.RFECV`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html) (only via `sklearn2pmml.SelectorProxy`)
    * [`feature_selection.SelectFdr`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFdr.html) (only via `sklearn2pmml.SelectorProxy`)
    * [`feature_selection.SelectFpr`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFpr.html) (only via `sklearn2pmml.SelectorProxy`)
    * [`feature_selection.SelectFromModel`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html) (either directly or via `sklearn2pmml.SelectorProxy`)
    * [`feature_selection.SelectFwe`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFwe.html) (only via `sklearn2pmml.SelectorProxy`)
    * [`feature_selection.SelectKBest`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html) (either directly or via `sklearn2pmml.SelectorProxy`)
    * [`feature_selection.SelectPercentile`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectPercentile.html) (only via `sklearn2pmml.SelectorProxy`)
    * [`feature_selection.VarianceThreshold`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.VarianceThreshold.html) (only via `sklearn2pmml.SelectorProxy`)
  * Impute:
    * [`impute.MissingIndicator`](https://scikit-learn.org/stable/modules/generated/sklearn.impute.MissingIndicator.html)
    * [`impute.SimpleImputer`](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html)
  * Isotonic regression:
    * [`isotonic.IsotonicRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.isotonic.IsotonicRegression.html)
  * Generalized Linear Models:
    * [`linear_model.ARDRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ARDRegression.html)
    * [`linear_model.BayesianRidge`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html)
    * [`linear_model.ElasticNet`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html)
    * [`linear_model.ElasticNetCV`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNetCV.html)
    * [`linear_model.GammaRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.GammaRegressor.html)
    * [`linear_model.HuberRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.HuberRegressor.html)
    * [`linear_model.Lars`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lars.html)
    * [`linear_model.LarsCV`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LarsCV.html)
    * [`linear_model.Lasso`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html)
    * [`linear_model.LassoCV`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html)
    * [`linear_model.LassoLars`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoLars.html)
    * [`linear_model.LassoLarsCV`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoLarsCV.html)
    * [`linear_model.LinearRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
    * [`linear_model.LogisticRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
    * [`linear_model.LogisticRegressionCV`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html)
    * [`linear_model.OrthogonalMatchingPursuit`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.OrthogonalMatchingPursuit.html)
    * [`linear_model.OrthogonalMatchingPursuitCV`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.OrthogonalMatchingPursuitCV.html)
    * [`linear_model.PoissonRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PoissonRegressor.html)
    * [`linear_model.Ridge`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)
    * [`linear_model.RidgeCV`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html)
    * [`linear_model.RidgeClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifier.html)
    * [`linear_model.RidgeClassifierCV`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifierCV.html)
    * [`linear_model.SGDClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html)
    * [`linear_model.SGDRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html)
    * [`linear_model.TheilSenRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.TheilSenRegressor.html)
  * Model Selection:
    * [`model_selection.GridSearchCV`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
    * [`model_selection.RandomizedSearchCV`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html)
  * Multiclass classification:
    * [`multiclass.OneVsRestClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html)
  * Naive Bayes:
    * [`naive_bayes.GaussianNB`](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html)
  * Nearest Neighbors:
    * [`neighbors.KNeighborsClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
    * [`neighbors.KNeighborsRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html)
  * Pipelines:
    * [`pipeline.FeatureUnion`](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.FeatureUnion.html)
    * [`pipeline.Pipeline`](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)
  * Neural network models:
    * [`neural_network.MLPClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)
    * [`neural_network.MLPRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html)
  * Preprocessing and Normalization:
    * [`preprocessing.Binarizer`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Binarizer.html)
    * [`preprocessing.FunctionTransformer`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.FunctionTransformer.html)
    * [`preprocessing.Imputer`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Imputer.html)
    * [`preprocessing.LabelBinarizer`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelBinarizer.html)
    * [`preprocessing.LabelEncoder`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html)
    * [`preprocessing.MaxAbsScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MaxAbsScaler.html)
    * [`preprocessing.MinMaxScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html)
    * [`preprocessing.OneHotEncoder`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)
    * [`preprocessing.OrdinalEncoder`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html)
    * [`preprocessing.PolynomialFeatures`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html)
    * [`preprocessing.RobustScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html)
    * [`preprocessing.StandardScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
  * Support Vector Machines:
    * [`svm.LinearSVC`](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html)
    * [`svm.LinearSVR`](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVR.html)
    * [`svm.OneClassSVM`](https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html)
    * [`svm.SVC`](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
    * [`svm.NuSVC`](https://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVC.html)
    * [`svm.SVR`](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html)
    * [`svm.NuSVR`](https://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVR.html)
  * Decision Trees:
    * [`tree.DecisionTreeClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
    * [`tree.DecisionTreeRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)
    * [`tree.ExtraTreeClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.tree.ExtraTreeClassifier.html)
    * [`tree.ExtraTreeRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.tree.ExtraTreeRegressor.html)
* Supported third-party Estimator and Transformer types:
  * [Category Encoders](http://contrib.scikit-learn.org/category_encoders/):
    * [`category_encoders.BaseNEncoder`](http://contrib.scikit-learn.org/category_encoders/basen.html)
    * [`category_encoders.BinaryEncoder`](http://contrib.scikit-learn.org/category_encoders/binary.html)
    * [`category_encoders.OrdinalEncoder`](http://contrib.scikit-learn.org/category_encoders/ordinal.html)
  * [H2O.ai](https://www.h2o.ai/):
    * [`h2o.estimators.gbm.H2OGradientBoostingEstimator`](http://docs.h2o.ai/h2o/latest-stable/h2o-py/docs/modeling.html#h2ogradientboostingestimator)
    * [`h2o.estimators.glm.H2OGeneralizedLinearEstimator`](http://docs.h2o.ai/h2o/latest-stable/h2o-py/docs/modeling.html#h2ogeneralizedlinearestimator)
    * [`h2o.estimators.isolation_forest.H2OIsolationForestEstimator`](http://docs.h2o.ai/h2o/latest-stable/h2o-py/docs/modeling.html#h2oisolationforestestimator)
    * [`h2o.estimators.random_forest.H2ORandomForestEstimator`](http://docs.h2o.ai/h2o/latest-stable/h2o-py/docs/modeling.html#h2orandomforestestimator)
    * [`h2o.estimators.stackedensemble.H2OStackedEnsembleEstimator`](http://docs.h2o.ai/h2o/latest-stable/h2o-py/docs/modeling.html#h2ostackedensembleestimator)
  * [Imbalanced-Learn (imblearn)](https://github.com/scikit-learn-contrib/imbalanced-learn):
    * [`imblearn.combine.SMOTEENN`](https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.combine.SMOTEENN.html)
    * [`imblearn.combine.SMOTETomek`](https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.combine.SMOTETomek.html)
    * [`imblearn.ensemble.BalancedBaggingClassifier`](https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.ensemble.BalancedBaggingClassifier.html)
    * [`imblearn,ensemble,BalancedRandomForestClassifier`](https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.ensemble.BalancedRandomForestClassifier.html)
    * [`imblearn.over_sampling.ADASYN`](https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.over_sampling.ADASYN.html)
    * [`imblearn.over_sampling.BorderlineSMOTE`](https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.over_sampling.BorderlineSMOTE.html)
    * [`imblearn.over_sampling.KMeansSMOTE`](https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.over_sampling.KMeansSMOTE.html)
    * [`imblearn.over_sampling.RandomOverSampler`](https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.over_sampling.RandomOverSampler.html)
    * [`imblearn.over_sampling.SMOTE`](https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.over_sampling.SMOTE.html)
    * [`imblearn.over_sampling.SMOTENC`](https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.over_sampling.SMOTENC.html)
    * [`imblearn.over_sampling.SVMSMOTE`](https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.over_sampling.SVMSMOTE.html)
    * [`imblearn.pipeline.Pipeline`](https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.pipeline.Pipeline.html)
    * [`imblearn.under_sampling.AllKNN`](https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.under_sampling.AllKNN.html)
    * [`imblearn.under_sampling.ClusterCentroids`](https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.under_sampling.ClusterCentroids.html)
    * [`imblearn.under_sampling.CondensedNearestNeighbour`](https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.under_sampling.CondensedNearestNeighbour.html)
    * [`imblearn.under_sampling.EditedNearestNeighbours`](https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.under_sampling.EditedNearestNeighbours.html)
    * [`imblearn.under_sampling.InstanceHardnessThreshold`](https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.under_sampling.InstanceHardnessThreshold.html)
    * [`imblearn.under_sampling.NearMiss`](https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.under_sampling.NearMiss.html)
    * [`imblearn.under_sampling.NeighbourhoodCleaningRule`](https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.under_sampling.NeighbourhoodCleaningRule.html)
    * [`imblearn.under_sampling.OneSidedSelection`](https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.under_sampling.OneSidedSelection.html)
    * [`imblearn.under_sampling.RandomUnderSampler`](https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.under_sampling.RandomUnderSampler.html)
    * [`imblearn.under_sampling.RepeatedEditedNearestNeighbours`](https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.under_sampling.RepeatedEditedNearestNeighbours.html)
    * [`imblearn.under_sampling.TomekLinks`](https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.under_sampling.TomekLinks.html)
  * [LightGBM](https://github.com/Microsoft/LightGBM):
    * [`lightgbm.LGBMClassifier`](https://lightgbm.readthedocs.io/en/latest/Python-API.html#lightgbm.LGBMClassifier)
    * [`lightgbm.LGBMRanker`](https://lightgbm.readthedocs.io/en/latest/Python-API.html#lightgbm.LGBMRanker)
    * [`lightgbm.LGBMRegressor`](https://lightgbm.readthedocs.io/en/latest/Python-API.html#lightgbm.LGBMRegressor)
  * [SkLearn2PMML](https://github.com/jpmml/sklearn2pmml):
    * `sklearn2pmml.EstimatorProxy`
    * `sklearn2pmml.SelectorProxy`
    * `sklearn2pmml.decoration.Alias`
    * `sklearn2pmml.decoration.CategoricalDomain`
    * `sklearn2pmml.decoration.ContinuousDomain`
    * `sklearn2pmml.decoration.DateDomain`
    * `sklearn2pmml.decoration.DateTimeDomain`
    * `sklearn2pmml.decoration.MultiDomain`
    * `sklearn2pmml.decoration.OrdinalDomain`
    * `sklearn2pmml.ensemble.GBDTLMRegressor`
      * The GBDT side: All Scikit-Learn decision tree ensemble regressors, `LGBMRegressor`, `XGBRegressor`, `XGBRFRegressor`.
      * The LM side: A Scikit-Learn linear regressor (eg. `ElasticNet`, `LinearRegression`, `SGDRegressor`).
    * `sklearn2pmml.ensemble.GBDTLRClassifier`
      * The GBDT side: All Scikit-Learn decision tree ensemble classifiers, `LGBMClassifier`, `XGBClassifier`, `XGBRFClassifier`.
      * The LR side: A Scikit-Learn binary linear classifier (eg. `LinearSVC`, `LogisticRegression`, `SGDClassifier`).
    * `sklearn2pmml.ensemble.SelectFirstClassifier`
    * `sklearn2pmml.ensemble.SelectFirstRegressor`
    * `sklearn2pmml.feature_selection.SelectUnique`
    * `sklearn2pmml.pipeline.PMMLPipeline`
    * `sklearn2pmml.preprocessing.Aggregator`
    * `sklearn2pmml.preprocessing.CastTransformer`
    * `sklearn2pmml.preprocessing.ConcatTransformer`
    * `sklearn2pmml.preprocessing.CutTransformer`
    * `sklearn2pmml.preprocessing.DaysSinceYearTransformer`
    * `sklearn2pmml.preprocessing.ExpressionTransformer`
      * Ternary conditional expression `<expression_true> if <condition> else <expression_false>`.
      * Array indexing expressions `X[<column index>]` and `X[<column name>]`.
      * String concatenation expressions.
      * Arithmetic operators `+`, `-`, `*`, `/` and `%`.
      * Identity comparison operators `is None` and `is not None`.
      * Comparison operators `in <list>`, `not in <list>`, `<=`, `<`, `==`, `!=`, `>` and `>=`.
      * Logical operators `and`, `or` and `not`.
      * Value missingness check functions `pandas.isnull` and `pandas.notnull`.
      * Numpy universal functions.
      * String functions `lower`, `upper` and `strip`.
    * `sklearn2pmml.preprocessing.IdentityTransformer`
    * `sklearn2pmml.preprocessing.LookupTransformer`
    * `sklearn2pmml.preprocessing.MatchesTransformer`
    * `sklearn2pmml.preprocessing.MultiLookupTransformer`
    * `sklearn2pmml.preprocessing.PMMLLabelBinarizer`
    * `sklearn2pmml.preprocessing.PMMLLabelEncoder`
    * `sklearn2pmml.preprocessing.PowerFunctionTransformer`
    * `sklearn2pmml.preprocessing.ReplaceTransformer`
    * `sklearn2pmml.preprocessing.SecondsSinceMidnightTransformer`
    * `sklearn2pmml.preprocessing.SecondsSinceYearTransformer`
    * `sklearn2pmml.preprocessing.StringNormalizer`
    * `sklearn2pmml.preprocessing.SubstringTransformer`
    * `sklearn2pmml.preprocessing.h2o.H2OFrameCreator`
    * `sklearn2pmml.preprocessing.scipy.BSplineTransformer`
    * `sklearn2pmml.ruleset.RuleSetClassifier`
  * [Sklearn-Pandas](https://github.com/paulgb/sklearn-pandas):
    * `sklearn_pandas.CategoricalImputer`
    * `sklearn_pandas.DataFrameMapper`
  * [TPOT](https://github.com/rhiever/tpot):
    * `tpot.builtins.stacking_estimator.StackingEstimator`
  * [XGBoost](https://github.com/dmlc/xgboost):
    * [`xgboost.XGBClassifier`](https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBClassifier)
    * [`xgboost.XGBRanker`](https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBRanker)
    * [`xgboost.XGBRegressor`](https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBRegressor)
    * [`xgboost.XGBRFClassifier`](https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBRFClassifier)
    * [`xgboost.XGBRFRegressor`](https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBRFRegressor)
* Production quality:
  * Complete test coverage.
  * Fully compliant with the [JPMML-Evaluator](https://github.com/jpmml/jpmml-evaluator) library.

# Prerequisites #

### The Python side of operations

* Python 2.7, 3.4 or newer.
* [`scikit-learn`](https://pypi.python.org/pypi/scikit-learn) 0.16.0 or newer.
* [`sklearn-pandas`](https://pypi.python.org/pypi/sklearn-pandas) 0.0.10 or newer.
* [`sklearn2pmml`](https://github.com/jpmml/sklearn2pmml) 0.14.0 or newer.

Validating Python installation:

```python
import sklearn, sklearn.externals.joblib, sklearn_pandas, sklearn2pmml

print(sklearn.__version__)
print(sklearn.externals.joblib.__version__)
print(sklearn_pandas.__version__)
print(sklearn2pmml.__version__)
```

### The JPMML-SkLearn side of operations

* Java 1.8 or newer.

# Installation #

Enter the project root directory and build using [Apache Maven](https://maven.apache.org/):
```
mvn clean install
```

The build produces an executable uber-JAR file `target/jpmml-sklearn-executable-1.6-SNAPSHOT.jar`.

# Usage #

A typical workflow can be summarized as follows:

1. Use Python to train a model.
2. Serialize the model in `pickle` data format to a file in a local filesystem.
3. Use the JPMML-SkLearn command-line converter application to turn the pickle file to a PMML file.

### The Python side of operations

Loading data to a `pandas.DataFrame` object:

```python
import pandas

df = pandas.read_csv("Iris.csv")

iris_X = df[df.columns.difference(["Species"])]
iris_y = df["Species"]
```

First, creating a `sklearn_pandas.DataFrameMapper` object, which performs **column-oriented** feature engineering and selection work:

```python
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import StandardScaler
from sklearn2pmml.decoration import ContinuousDomain

column_preprocessor = DataFrameMapper([
    (["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"], [ContinuousDomain(), StandardScaler()])
])
```

Second, creating `Transformer` and `Selector` objects, which perform **table-oriented** feature engineering and selection work:

```python
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline
from sklearn2pmml import SelectorProxy

table_preprocessor = Pipeline([
	("pca", PCA(n_components = 3)),
	("selector", SelectorProxy(SelectKBest(k = 2)))
])
```

Please note that stateless Scikit-Learn selector objects need to be wrapped into an `sklearn2pmml.SelectprProxy` object.

Third, creating an `Estimator` object:

```python
from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier(min_samples_leaf = 5)
```

Combining the above objects into a `sklearn2pmml.pipeline.PMMLPipeline` object, and running the experiment:

```python
from sklearn2pmml.pipeline import PMMLPipeline

pipeline = PMMLPipeline([
    ("columns", column_preprocessor),
    ("table", table_preprocessor),
    ("classifier", classifier)
])
pipeline.fit(iris_X, iris_y)
```

Embedding model verification data:

```python
pipeline.verify(iris_X.sample(n = 15))
```

Storing the fitted `PMMLPipeline` object in `pickle` data format:

```python
from sklearn.externals import joblib

joblib.dump(pipeline, "pipeline.pkl.z", compress = 9)
```

Please see the test script file [main.py](https://github.com/jpmml/jpmml-sklearn/blob/master/src/test/resources/main.py) for more classification (binary and multi-class) and regression workflows.

### The JPMML-SkLearn side of operations

Converting the pipeline pickle file `pipeline.pkl.z` to a PMML file `pipeline.pmml`:
```
java -jar target/jpmml-sklearn-executable-1.6-SNAPSHOT.jar --pkl-input pipeline.pkl.z --pmml-output pipeline.pmml
```

Getting help:
```
java -jar target/jpmml-sklearn-executable-1.6-SNAPSHOT.jar --help
```

# Documentation #

Up-to-date:

* [Converting Scikit-Learn based Imbalanced-Learn (imblearn) pipelines to PMML documents](https://openscoring.io/blog/2020/10/24/converting_sklearn_imblearn_pipeline_pmml/)
* [Extending Scikit-Learn with date and datetime features](https://openscoring.io/blog/2020/03/08/sklearn_date_datetime_pmml/)
* [Extending Scikit-Learn with feature specifications](https://openscoring.io/blog/2020/02/23/sklearn_feature_specification_pmml/)
* [Converting logistic regression models to PMML documents](https://openscoring.io/blog/2020/01/19/converting_logistic_regression_pmml/)
* [Stacking Scikit-Learn, LightGBM and XGBoost models](https://openscoring.io/blog/2020/01/02/stacking_sklearn_lightgbm_xgboost/)
* [Converting Scikit-Learn hyperparameter-tuned pipelines to PMML documents](https://openscoring.io/blog/2019/12/25/converting_sklearn_gridsearchcv_pipeline_pmml/)
* [Extending Scikit-Learn with GBDT plus LR ensemble (GBDT+LR) model type](https://openscoring.io/blog/2019/06/19/sklearn_gbdt_lr_ensemble/)
* [Converting Scikit-Learn based TPOT automated machine learning (AutoML) pipelines to PMML documents](https://openscoring.io/blog/2019/06/10/converting_sklearn_tpot_pipeline_pmml/)
* [Converting Scikit-Learn based LightGBM pipelines to PMML documents](https://openscoring.io/blog/2019/04/07/converting_sklearn_lightgbm_pipeline_pmml/)
* [Extending Scikit-Learn with business rules (BR) model type](https://openscoring.io/blog/2018/09/17/sklearn_business_rules/)

Slightly outdated:

* [Converting Scikit-Learn to PMML](https://www.slideshare.net/VilluRuusmann/converting-scikitlearn-to-pmml)

# License #

JPMML-SkLearn is licensed under the terms and conditions of the [GNU Affero General Public License, Version 3.0](https://www.gnu.org/licenses/agpl-3.0.html).

If you would like to use JPMML-SkLearn in a proprietary software project, then it is possible to enter into a licensing agreement which makes JPMML-SkLearn available under the terms and conditions of the [BSD 3-Clause License](https://opensource.org/licenses/BSD-3-Clause) instead.

# Additional information #

JPMML-SkLearn is developed and maintained by Openscoring Ltd, Estonia.

Interested in using [Java PMML API](https://github.com/jpmml) software in your company? Please contact [info@openscoring.io](mailto:info@openscoring.io)
