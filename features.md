<details>
  <summary>Scikit-Learn</summary>

  Examples: [main.py](https://github.com/jpmml/jpmml-sklearn/blob/master/pmml-sklearn/src/test/resources/main.py)

  * Probability Calibration:
    * [`calibration.CalibratedClassifierCV`](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html)
  * Clustering:
    * [`cluster.KMeans`](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
    * [`cluster.MiniBatchKMeans`](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html)
  * Composite estimators:
    * [`compose.ColumnTransformer`](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html)
    * [`compose.TransformedTargetRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.compose.TransformedTargetRegressor.html)
  * Matrix decomposition:
    * [`decomposition.PCA`](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
    * [`decomposition.IncrementalPCA`](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.IncrementalPCA.html)
    * [`decomposition.TruncatedSVD`](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html)
  * Discriminant analysis:
    * [`discriminant_analysis.LinearDiscriminantAnalysis`](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html)
  * Dummies:
    * [`dummy.DummyClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html)
    * [`dummy.DummyRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyRegressor.html)
  * Ensemble methods:
    * [`ensemble.AdaBoostClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html)
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
  * Feature extraction:
    * [`feature_extraction.DictVectorizer`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.DictVectorizer.html)
    * [`feature_extraction.text.CountVectorizer`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)
    * [`feature_extraction.text.TfidfVectorizer`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
  * Feature selection:
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
  * Freezing:
    * [`frozen.FrozenEstimator`](https://scikit-learn.org/stable/modules/generated/sklearn.frozen.FrozenEstimator.html)
  * Impute:
    * [`impute.MissingIndicator`](https://scikit-learn.org/stable/modules/generated/sklearn.impute.MissingIndicator.html)
    * [`impute.SimpleImputer`](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html)
  * Isotonic regression:
    * [`isotonic.IsotonicRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.isotonic.IsotonicRegression.html)
  * Kernel ridge regression:
    * [`kernel_ridge.KernelRidge`](https://scikit-learn.org/stable/modules/generated/sklearn.kernel_ridge.KernelRidge.html)
  * Generalized linear models:
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
    * [`linear_model.Perceptron`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html)
    * [`linear_model.PoissonRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PoissonRegressor.html)
    * [`linear_model.QuantileRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.QuantileRegressor.html)
    * [`linear_model.Ridge`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)
    * [`linear_model.RidgeCV`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html)
    * [`linear_model.RidgeClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifier.html)
    * [`linear_model.RidgeClassifierCV`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifierCV.html)
    * [`linear_model.SGDClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html)
    * [`linear_model.SGDOneClassSVM`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDOneClassSVM.html)
    * [`linear_model.SGDRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html)
    * [`linear_model.TheilSenRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.TheilSenRegressor.html)
    * [`linear_model.TweedieRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.TweedieRegressor.html)
  * Model selection:
    * [`model_selection.GridSearchCV`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
    * [`model_selection.RandomizedSearchCV`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html)
  * Post-fit Model tuning:
    * [`model_selection.FixedThresholdClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.FixedThresholdClassifier.html)
    * [`model_selection.TunedThresholdClassifierCV`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TunedThresholdClassifierCV.html)
  * Multiclass classification:
    * [`multiclass.OneVsRestClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html)
  * Multioutput regression and classification:
    * [`multioutput.ClassifierChain`](https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.ClassifierChain.html)
    * [`multioutput.MultiOutputClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html)
    * [`multioutput.MultiOutputRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputRegressor.html)
    * [`multioutput.RegressorChain`](https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.RegressorChain.html)
  * Naive Bayes:
    * [`naive_bayes.CategoricalNB`](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.CategoricalNB.html)
    * [`naive_bayes.BernoulliNB`](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html)
    * [`naive_bayes.GaussianNB`](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html)
    * [`naive_bayes.MultinomialNB`](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html)
  * Nearest neighbors:
    * [`neighbors.KNeighborsClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
    * [`neighbors.KNeighborsRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html)
    * [`neighbors.NearestCentroid`](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestCentroid.html)
    * [`neighbors.NearestNeighbors`](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html)
  * Pipelines:
    * [`pipeline.FeatureUnion`](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.FeatureUnion.html)
    * [`pipeline.Pipeline`](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)
  * Neural network models:
    * [`neural_network.MLPClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)
    * [`neural_network.MLPRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html)
  * Preprocessing and normalization:
    * [`preprocessing.Binarizer`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Binarizer.html)
    * [`preprocessing.FunctionTransformer`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.FunctionTransformer.html)
    * [`preprocessing.Imputer`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Imputer.html)
    * [`preprocessing.KBinsDiscretizer`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.KBinsDiscretizer.html)
    * [`preprocessing.LabelBinarizer`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelBinarizer.html)
    * [`preprocessing.LabelEncoder`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html)
    * [`preprocessing.MaxAbsScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MaxAbsScaler.html)
    * [`preprocessing.MinMaxScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html)
    * [`preprocessing.Normalizer`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Normalizer.html)
    * [`preprocessing.OneHotEncoder`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)
    * [`preprocessing.OrdinalEncoder`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html)
    * [`preprocessing.PolynomialFeatures`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html)
    * [`preprocessing.PowerTransformer`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PowerTransformer.html)
    * [`preprocessing.QuantileTransformer`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html)
    * [`preprocessing.RobustScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html)
    * [`preprocessing.SplineTransformer`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.SplineTransformer.html)
    * [`preprocessing.StandardScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
    * [`preprocessing.TargetEncoder`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.TargetEncoder.html)
  * Support vector machines:
    * [`svm.LinearSVC`](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html)
    * [`svm.LinearSVR`](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVR.html)
    * [`svm.OneClassSVM`](https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html)
    * [`svm.SVC`](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
    * [`svm.NuSVC`](https://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVC.html)
    * [`svm.SVR`](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html)
    * [`svm.NuSVR`](https://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVR.html)
  * Decision trees:
    * [`tree.DecisionTreeClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
    * [`tree.DecisionTreeRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)
    * [`tree.ExtraTreeClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.tree.ExtraTreeClassifier.html)
    * [`tree.ExtraTreeRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.tree.ExtraTreeRegressor.html)
</details>

<details>
  <summary>BorutaPy</summary>

  Examples: [extensions/boruta.py](https://github.com/jpmml/jpmml-sklearn/blob/master/pmml-sklearn-extension/src/test/resources/extensions/boruta.py)

  * `boruta.BorutaPy`
</details>

<details>
  <summary>Category Encoders</summary>

  Examples: [extensions/category_encoders.py](https://github.com/jpmml/jpmml-sklearn/blob/master/pmml-sklearn-extension/src/test/resources/extensions/category_encoders.py) and [extensions/category_encoders-xgboost.py](https://github.com/jpmml/jpmml-sklearn/blob/master/pmml-sklearn-xgboost/src/test/resources/extensions/category_encoders-xgboost.py)

  * [`category_encoders.BaseNEncoder`](https://contrib.scikit-learn.org/category_encoders/basen.html)
  * [`category_encoders.BinaryEncoder`](https://contrib.scikit-learn.org/category_encoders/binary.html)
  * [`category_encoders.CatBoostEncoder`](https://contrib.scikit-learn.org/category_encoders/catboost.html)
  * [`category_encoders.CountEncoder`](https://contrib.scikit-learn.org/category_encoders/count.html)
  * [`category_encoders.LeaveOneOutEncoder`](https://contrib.scikit-learn.org/category_encoders/leaveoneout.html)
  * [`category_encoders.OneHotEncoder`](https://contrib.scikit-learn.org/category_encoders/onehot.html)
  * [`category_encoders.OrdinalEncoder`](https://contrib.scikit-learn.org/category_encoders/ordinal.html)
  * [`category_encoders.TargetEncoder`](https://contrib.scikit-learn.org/category_encoders/targetencoder.html)
  * [`category_encoders.WOEEncoder`](https://contrib.scikit-learn.org/category_encoders/woe.html)
</details>

<details>
  <summary>FLAML</summary>

  Examples: [extensions/flaml.py](https://github.com/jpmml/jpmml-sklearn/blob/master/pmml-sklearn-extension/src/test/resources/extensions/flaml.py)

  * `flaml.automl.contrib.histgb.HistGradientBoostingEstimator`
  * `flaml.automl.model.ElasticNetEstimator`
  * `flaml.automl.model.ExtraTreesEstimator`
  * `flaml.automl.model.LassoLarsEstimator`
  * `flaml.automl.model.LGBMEstimator`
  * `flaml.automl.model.LRL1Classifier`
  * `flaml.automl.model.LRL2Classifier`
  * `flaml.automl.model.RandomForestEstimator`
  * `flaml.automl.model.SGDEstimator`
  * `flaml.automl.model.SVCEstimator`
  * `flaml.automl.model.XGBoostLimitDepthEstimator`
  * `flaml.automl.model.XGBoostSklearnEstimator`
</details>

<details>
  <summary>H2O.ai</summary>

  Examples: [main-h2o.py](https://github.com/jpmml/jpmml-sklearn/blob/master/pmml-sklearn-h2o/src/test/resources/main-h2o.py)

  * [`h2o.estimators.extended_isolation_forest.H2OExtendedIsolationForestEstimator`](https://docs.h2o.ai/h2o/latest-stable/h2o-py/docs/modeling.html#h2oextendedisolationforestestimator)
  * [`h2o.estimators.gbm.H2OGradientBoostingEstimator`](https://docs.h2o.ai/h2o/latest-stable/h2o-py/docs/modeling.html#h2ogradientboostingestimator)
  * [`h2o.estimators.glm.H2OGeneralizedLinearEstimator`](https://docs.h2o.ai/h2o/latest-stable/h2o-py/docs/modeling.html#h2ogeneralizedlinearestimator)
  * [`h2o.estimators.isolation_forest.H2OIsolationForestEstimator`](https://docs.h2o.ai/h2o/latest-stable/h2o-py/docs/modeling.html#h2oisolationforestestimator)
  * [`h2o.estimators.random_forest.H2ORandomForestEstimator`](https://docs.h2o.ai/h2o/latest-stable/h2o-py/docs/modeling.html#h2orandomforestestimator)
  * [`h2o.estimators.stackedensemble.H2OStackedEnsembleEstimator`](https://docs.h2o.ai/h2o/latest-stable/h2o-py/docs/modeling.html#h2ostackedensembleestimator)
  * [`h2o.estimators.xgboost.H2OXGBoostEstimator`](https://docs.h2o.ai/h2o/latest-stable/h2o-py/docs/modeling.html#h2oxgboostestimator)
</details>

<details>
  <summary>Hyperopt-sklearn</summary>

  Examples: [extensions/hpsklearn.py](https://github.com/jpmml/jpmml-sklearn/blob/master/pmml-sklearn-extension/src/test/resources/extensions/hpsklearn.py)

  * `hpsklearn.HyperoptEstimator`
</details>

<details>
  <summary>Imbalanced-Learn</summary>

  Examples: [extensions/imblearn.py](https://github.com/jpmml/jpmml-sklearn/blob/master/pmml-sklearn-extension/src/test/resources/extensions/imblearn.py)

  * Under-sampling methods:
    * [`imblearn.under_sampling.AllKNN`](https://imbalanced-learn.org/stable/references/generated/imblearn.under_sampling.AllKNN.html)
    * [`imblearn.under_sampling.ClusterCentroids`](https://imbalanced-learn.org/stable/references/generated/imblearn.under_sampling.ClusterCentroids.html)
    * [`imblearn.under_sampling.CondensedNearestNeighbour`](https://imbalanced-learn.org/stable/references/generated/imblearn.under_sampling.CondensedNearestNeighbour.html)
    * [`imblearn.under_sampling.EditedNearestNeighbours`](https://imbalanced-learn.org/stable/references/generated/imblearn.under_sampling.EditedNearestNeighbours.html)
    * [`imblearn.under_sampling.InstanceHardnessThreshold`](https://imbalanced-learn.org/stable/references/generated/imblearn.under_sampling.InstanceHardnessThreshold.html)
    * [`imblearn.under_sampling.NearMiss`](https://imbalanced-learn.org/stable/references/generated/imblearn.under_sampling.NearMiss.html)
    * [`imblearn.under_sampling.NeighbourhoodCleaningRule`](https://imbalanced-learn.org/stable/references/generated/imblearn.under_sampling.NeighbourhoodCleaningRule.html)
    * [`imblearn.under_sampling.OneSidedSelection`](https://imbalanced-learn.org/stable/references/generated/imblearn.under_sampling.OneSidedSelection.html)
    * [`imblearn.under_sampling.RandomUnderSampler`](https://imbalanced-learn.org/stable/references/generated/imblearn.under_sampling.RandomUnderSampler.html)
    * [`imblearn.under_sampling.RepeatedEditedNearestNeighbours`](https://imbalanced-learn.org/stable/references/generated/imblearn.under_sampling.RepeatedEditedNearestNeighbours.html)
    * [`imblearn.under_sampling.TomekLinks`](https://imbalanced-learn.org/stable/references/generated/imblearn.under_sampling.TomekLinks.html)
  * Over-sampling methods:
    * [`imblearn.over_sampling.ADASYN`](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.ADASYN.html)
    * [`imblearn.over_sampling.BorderlineSMOTE`](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.BorderlineSMOTE.html)
    * [`imblearn.over_sampling.KMeansSMOTE`](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.KMeansSMOTE.html)
    * [`imblearn.over_sampling.RandomOverSampler`](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.RandomOverSampler.html)
    * [`imblearn.over_sampling.SMOTE`](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html)
    * [`imblearn.over_sampling.SMOTENC`](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTENC.html)
    * [`imblearn.over_sampling.SVMSMOTE`](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SVMSMOTE.html)
  * Combination of over- and under-sampling methods:
    * [`imblearn.combine.SMOTEENN`](https://imbalanced-learn.org/stable/references/generated/imblearn.combine.SMOTEENN.html)
    * [`imblearn.combine.SMOTETomek`](https://imbalanced-learn.org/stable/references/generated/imblearn.combine.SMOTETomek.html)
  * Ensemble methods:
    * [`imblearn.ensemble.BalancedBaggingClassifier`](https://imbalanced-learn.org/stable/references/generated/imblearn.ensemble.BalancedBaggingClassifier.html)
    * [`imblearn,ensemble,BalancedRandomForestClassifier`](https://imbalanced-learn.org/stable/references/generated/imblearn.ensemble.BalancedRandomForestClassifier.html)
  * Pipeline:
    * [`imblearn.pipeline.Pipeline`](https://imbalanced-learn.org/stable/references/generated/imblearn.pipeline.Pipeline.html)
</details>

<details>
  <summary>InterpretML</summary>

  Examples: [extensions/interpret.py](https://github.com/jpmml/jpmml-sklearn/blob/master/pmml-sklearn-extension/src/test/resources/extensions/interpret.py)

  * [`interpret.glassbox.ClassificationTree`](https://interpret.ml/docs/python/api/ClassificationTree.html)
  * [`interpret.glassbox.ExplainableBoostingClassifier`](https://interpret.ml/docs/python/api/ExplainableBoostingClassifier.html)
  * [`interpret.glassbox.ExplainableBoostingRegressor`](https://interpret.ml/docs/python/api/ExplainableBoostingRegressor.html)
  * [`interpret.glassbox.LinearRegression`](https://interpret.ml/docs/python/api/LinearRegression.html)
  * [`interpret.glassbox.LogisticRegression`](https://interpret.ml/docs/python/api/LogisticRegression.html)
  * [`interpret.glassbox.RegressionTree`](https://interpret.ml/docs/python/api/RegressionTree.html)
</details>

<details>
  <summary>LightGBM</summary>

  Examples: [main-lightgbm.py](https://github.com/jpmml/jpmml-sklearn/blob/master/pmml-sklearn-lightgbm/src/test/resources/main-lightgbm.py)

  * [`lightgbm.Booster`](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.Booster.html)
  * [`lightgbm.LGBMClassifier`](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMClassifier.html)
  * [`lightgbm.LGBMRanker`](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMRanker.html)
  * [`lightgbm.LGBMRegressor`](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMRegressor.html)
</details>

<details>
  <summary>Mlxtend</summary>

  Examples: N/A

  * [`mlxtend.preprocessing.DenseTransformer`](https://rasbt.github.io/mlxtend/user_guide/preprocessing/DenseTransformer/)
</details>

<details>
  <summary>NGBoost</summary>

  Examples: [extensions/ngboost.py](https://github.com/jpmml/jpmml-sklearn/blob/master/pmml-sklearn-extension/src/test/resources/extensions/ngboost.py)

  * `ngboost.NGBClassifier`
  * `ngboost.NGBRegressor`
</details>

<details>
  <summary>OptBinning</summary>

  Examples: [extensions/optbinning.py](https://github.com/jpmml/jpmml-sklearn/blob/master/pmml-sklearn-extension/src/test/resources/extensions/optbinning.py)

  * [`optbinning.BinningProcess`](https://gnpalencia.org/optbinning/binning_process.html#optbinning.BinningProcess)
  * [`optbinning.ContinuousOptimalBinning`](https://gnpalencia.org/optbinning/binning_continuous.html#optbinning.ContinuousOptimalBinning)
  * [`optbinning.MulticlassOptimalBinning`](https://gnpalencia.org/optbinning/binning_multiclass.html#optbinning.MulticlassOptimalBinning)
  * [`optbinning.OptimalBinning`](https://gnpalencia.org/optbinning/binning_binary.html#optbinning.OptimalBinning)
  * [`optbinning.scorecard.Scorecard`](https://gnpalencia.org/optbinning/scorecard.html#optbinning.scorecard.Scorecard)
</details>

<details>
  <summary>PyCaret</summary>

  Examples: [extensions/pycaret.py](https://github.com/jpmml/jpmml-sklearn/blob/master/pmml-sklearn-extension/src/test/resources/extensions/pycaret.py)

  * `pycaret.internal.pipeline.Pipeline`
  * `pycaret.internal.preprocess.transformers.CleanColumnNames`
  * `pycaret.internal.preprocess.transformers.FixImbalancer`
  * `pycaret.internal.preprocess.transformers.RareCategoryGrouping`
  * `pycaret.internal.preprocess.transformers.RemoveMulticollinearity`
  * `pycaret.internal.preprocess.transformers.RemoveOutliers`
  * `pycaret.internal.preprocess.transformers.TransformerWrapper`
  * `pycaret.internal.preprocess.transformers.TransformerWrapperWithInverse`
</details>

<details>
  <summary>Scikit-Lego</summary>

  Examples: [extensions/sklego.py](https://github.com/jpmml/jpmml-sklearn/blob/master/pmml-sklearn-extension/src/test/resources/extensions/sklego.py)

  * [`sklego.meta.EstimatorTransformer`](https://koaning.github.io/scikit-lego/api/meta/#sklego.meta.estimator_transformer.EstimatorTransformer)
    * Predict functions `apply`, `decision_function`, `predict` and `predict_proba`.
  * [`sklego.meta.OrdinalClassifier`](https://koaning.github.io/scikit-lego/api/meta/#sklego.meta.ordinal_classification.OrdinalClassifier)
  * [`sklego.pipeline.DebugPipeline`](https://koaning.github.io/scikit-lego/api/pipeline/#sklego.pipeline.DebugPipeline)
  * [`sklego.preprocessing.IdentityTransformer`](https://koaning.github.io/scikit-lego/api/preprocessing/#sklego.preprocessing.identitytransformer.IdentityTransformer)
</details>

<details>
  <summary>SkLearn2PMML</summary>

  Examples: [main.py](https://github.com/jpmml/jpmml-sklearn/blob/master/pmml-sklearn/src/test/resources/main.py) and [extensions/sklearn2pmml.py](https://github.com/jpmml/jpmml-sklearn/blob/master/pmml-sklearn/src/test/resources/extensions/sklearn2pmml.py)

  * Helpers:
    * `sklearn2pmml.EstimatorProxy`
    * `sklearn2pmml.SelectorProxy`
    * `sklearn2pmml.h2o.H2OEstimatorProxy`
  * Feature cross-references:
    * `sklearn2pmml.cross_reference.Memorizer`
    * `sklearn2pmml.cross_reference.Recaller`
  * Feature specification and decoration:
    * `sklearn2pmml.decoration.Alias`
    * `sklearn2pmml.decoration.CategoricalDomain`
    * `sklearn2pmml.decoration.ContinuousDomain`
    * `sklearn2pmml.decoration.ContinuousDomainEraser`
    * `sklearn2pmml.decoration.DateDomain`
    * `sklearn2pmml.decoration.DateTimeDomain`
    * `sklearn2pmml.decoration.DiscreteDomainEraser`
    * `sklearn2pmml.decoration.MultiAlias`
    * `sklearn2pmml.decoration.MultiDomain`
    * `sklearn2pmml.decoration.OrdinalDomain`
  * Ensemble methods:
    * `sklearn2pmml.ensemble.EstimatorChain`
    * `sklearn2pmml.ensemble.GBDTLMRegressor`
      * The GBDT side: All Scikit-Learn decision tree ensemble regressors, `LGBMRegressor`, `XGBRegressor`, `XGBRFRegressor`.
      * The LM side: A Scikit-Learn linear regressor (eg. `ElasticNet`, `LinearRegression`, `SGDRegressor`).
    * `sklearn2pmml.ensemble.GBDTLRClassifier`
      * The GBDT side: All Scikit-Learn decision tree ensemble classifiers, `LGBMClassifier`, `XGBClassifier`, `XGBRFClassifier`.
      * The LR side: A Scikit-Learn binary linear classifier (eg. `LinearSVC`, `LogisticRegression`, `SGDClassifier`).
    * `sklearn2pmml.ensemble.SelectFirstClassifier`
    * `sklearn2pmml.ensemble.SelectFirstRegressor`
  * UDF models:
    * `sklearn2pmml.expression.ExpressionClassifier`
    * `sklearn2pmml.expression.ExpressionRegressor`
  * Feature selection:
    * `sklearn2pmml.feature_selection.SelectUnique`
  * Linear models:
    * `sklearn2pmml.statsmodels.StatsModelsClassifier`
    * `sklearn2pmml.statsmodels.StatsModelsOrdinalClassifier`
    * `sklearn2pmml.statsmodels.StatsModelsRegressor`
  * Neural networks:
    * `sklearn2pmml.neural_network.MLPTransformer`
  * Pipeline:
    * `sklearn2pmml.pipeline.PMMLPipeline`
  * Postprocessing:
    * `sklearn2pmml.postprocessing.BusinessDecisionTransformer`
    * `sklearn2pmml.postprocessing.FeatureExporter`
  * Preprocessing:
    * `sklearn2pmml.preprocessing.AggregateTransformer`
    * `sklearn2pmml.preprocessing.BSplineTransformer`
    * `sklearn2pmml.preprocessing.CastTransformer`
    * `sklearn2pmml.preprocessing.ConcatTransformer`
    * `sklearn2pmml.preprocessing.CutTransformer`
    * `sklearn2pmml.preprocessing.DataFrameConstructor`
    * `sklearn2pmml.preprocessing.DateTimeFormatter`
    * `sklearn2pmml.preprocessing.DaysSinceYearTransformer`
    * `sklearn2pmml.preprocessing.ExpressionTransformer`
      * Ternary conditional expression `<expression_true> if <condition> else <expression_false>`.
      * Array indexing expressions `X[<column index>]` and `X[<column name>]`.
      * String concatenation expressions.
      * String slicing expressions `<str>[<start>:<stop>]`.
      * Arithmetic operators `+`, `-`, `*`, `/`, `//` and `%`.
      * The power operator `**`.
      * Identity comparison operators `is None` and `is not None`.
      * Comparison operators `in <list>`, `not in <list>`, `<=`, `<`, `==`, `!=`, `>` and `>=`.
      * Logical operators `and`, `or` and `not`.
      * Built-in functions (too numerous to list).
      * Built-in type cast functions `bool`, `float`, `int` and `str`.
      * Math constants `math.e`, `math.nan`, `math.pi` and `math.tau`.
      * Math functions (too numerous to list).
      * Numpy constants `numpy.e`, `numpy.NaN`. `numpy.NZERO`, `numpy.pi` and `numpy.PZERO`.
      * Numpy function `numpy.where`.
      * Numpy universal functions (too numerous to list).
      * Pandas constants `pandas.NA` and `pandas.NaT`.
      * Pandas functions `pandas.isna`, `pandas.isnull`, `pandas.notna` and `pandas.notnull`.
      * Scipy functions `scipy.special.expit` and `scipy.special.logit`.
      * String functions `startswith(<prefix>)`, `endswith(<suffix>)`, `lower`, `upper` and `strip`.
      * String length function `len(<str>)`.
      * Perl Compatible Regular Expression (PCRE) functions `pcre.search` and `pcre.sub`.
      * Regular Expression (RE) functions `re.search`, and `re.sub`.
      * User-defined functions.
    * `sklearn2pmml.preprocessing.FilterLookupTransformer`
    * `sklearn2pmml.preprocessing.IdentityTransformer`
    * `sklearn2pmml.preprocessing.LagTransformer`
    * `sklearn2pmml.preprocessing.LookupTransformer`
    * `sklearn2pmml.preprocessing.MatchesTransformer`
    * `sklearn2pmml.preprocessing.MultiCastTransformer`
    * `sklearn2pmml.preprocessing.MultiLookupTransformer`
    * `sklearn2pmml.preprocessing.NumberFormatter`
    * `sklearn2pmml.preprocessing.PMMLLabelBinarizer`
    * `sklearn2pmml.preprocessing.PMMLLabelEncoder`
    * `sklearn2pmml.preprocessing.PowerFunctionTransformer`
    * `sklearn2pmml.preprocessing.ReplaceTransformer`
    * `sklearn2pmml.preprocessing.RollingAggregateTransformer`
    * `sklearn2pmml.preprocessing.SecondsSinceMidnightTransformer`
    * `sklearn2pmml.preprocessing.SecondsSinceYearTransformer`
    * `sklearn2pmml.preprocessing.SelectFirstTransformer`
    * `sklearn2pmml.preprocessing.SeriesConstructor`
    * `sklearn2pmml.preprocessing.StringLengthTransformer`
    * `sklearn2pmml.preprocessing.StringNormalizer`
    * `sklearn2pmml.preprocessing.SubstringTransformer`
    * `sklearn2pmml.preprocessing.WordCountTransformer`
    * `sklearn2pmml.preprocessing.h2o.H2OFrameConstructor`
    * `sklearn2pmml.util.Reshaper`
    * `sklearn2pmml.util.Slicer`
  * Rule sets:
    * `sklearn2pmml.ruleset.RuleSetClassifier`
  * Decision trees:
    * `sklearn2pmml.tree.chaid.CHAIDClassifier`
    * `sklearn2pmml.tree.chaid.CHAIDRegressor`
</details>

<details>
  <summary>Sklearn-Pandas</summary>

  Examples: [main.py](https://github.com/jpmml/jpmml-sklearn/blob/master/pmml-sklearn/src/test/resources/main.py)

  * `sklearn_pandas.CategoricalImputer`
  * `sklearn_pandas.DataFrameMapper`
</details>

<details>
  <summary>StatsModels</summary>

  Examples: [main-statsmodels.py](https://github.com/jpmml/jpmml-sklearn/blob/master/pmml-sklearn-statsmodels/src/test/resources/main-statsmodels.py)

  * [`statsmodels.api.GLM`](https://www.statsmodels.org/stable/generated/statsmodels.genmod.generalized_linear_model.GLM.html)
  * [`statsmodels.api.Logit`](https://www.statsmodels.org/dev/generated/statsmodels.discrete.discrete_model.Logit.html)
  * [`statsmodels.api.MNLogit`](https://www.statsmodels.org/dev/generated/statsmodels.discrete.discrete_model.MNLogit.html)
  * [`statsmodels.api.OLS`](https://www.statsmodels.org/dev/generated/statsmodels.regression.linear_model.OLS.html)
  * [`statsmodels.api.Poisson`](https://www.statsmodels.org/dev/generated/statsmodels.discrete.discrete_model.Poisson.html)
  * [`statsmodels.api.QuantReg`](https://www.statsmodels.org/dev/generated/statsmodels.regression.quantile_regression.QuantReg.html)
  * [`statsmodels.api.WLS`](https://www.statsmodels.org/dev/generated/statsmodels.regression.linear_model.WLS.html)
  * [`statsmodels.miscmodels.ordinal_model.OrderedModel`](https://www.statsmodels.org/dev/generated/statsmodels.miscmodels.ordinal_model.OrderedModel.html)
</details>

<details>
  <summary>TPOT</summary>

  Examples: [extensions/tpot.py](https://github.com/jpmml/jpmml-sklearn/blob/master/pmml-sklearn-extension/src/test/resources/extensions/tpot.py)

  * `tpot.builtins.stacking_estimator.StackingEstimator`
</details>

<details>
  <summary>Treeple (formerly Scikit-Tree)</summary>

  Examples: [extensions/treeple.py](https://github.com/jpmml/jpmml-sklearn/blob/master/pmml-sklearn-extension/src/test/resources/extensions/treeple.py)

  * [`treeple.ExtendedIsolationForest`](https://docs.neurodata.io/treeple/dev/generated/treeple.ExtendedIsolationForest.html)
  * [`treeple.ObliqueRandomForestClassifier`](https://docs.neurodata.io/treeple/dev/generated/treeple.ObliqueRandomForestClassifier.html)
  * [`treeple.ObliqueRandomForestRegressor`](https://docs.neurodata.io/treeple/dev/generated/treeple.ObliqueRandomForestRegressor.html)
  * [`treeple.tree.ObliqueDecisionTreeClassifier`](https://docs.neurodata.io/treeple/dev/generated/treeple.tree.ObliqueDecisionTreeClassifier.html)
  * [`treeple.tree.ObliqueDecisionTreeRegressor`](https://docs.neurodata.io/treeple/dev/generated/treeple.tree.ObliqueDecisionTreeRegressor.html)
</details>

<details>
  <summary>XGBoost</summary>

  Examples: [main-xgboost.py](https://github.com/jpmml/jpmml-sklearn/blob/master/pmml-sklearn-xgboost/src/test/resources/main-xgboost.py), [extensions/category_encoders-xgboost.py](https://github.com/jpmml/jpmml-sklearn/blob/master/pmml-sklearn-xgboost/src/test/resources/extensions/category_encoders-xgboost.py) and [extensions/categorical.py](https://github.com/jpmml/jpmml-sklearn/blob/master/pmml-sklearn-xgboost/src/test/resources/extensions/categorical.py)

  * [`gbboost.Booster`](https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.Booster)
  * [`xgboost.XGBClassifier`](https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBClassifier)
  * [`xgboost.XGBRanker`](https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBRanker)
  * [`xgboost.XGBRegressor`](https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBRegressor)
  * [`xgboost.XGBRFClassifier`](https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBRFClassifier)
  * [`xgboost.XGBRFRegressor`](https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBRFRegressor)
</details>