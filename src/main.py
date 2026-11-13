from ucimlrepo import fetch_ucirepo 
import pandas
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
from sklearn.metrics import average_precision_score, f1_score
  
# fetch dataset 
mushroom = fetch_ucirepo(id=73)
dataset = mushroom.data
feature_list = dataset.features.columns.tolist()

#data audit
print(mushroom.metadata,"\n", mushroom.variables, f"Features: {feature_list}\n", f"Data Shape: {dataset.features.shape}\n", f"Model Target: {dataset.targets.columns.tolist()}\n")
print(pandas.concat((dataset.features,dataset.targets),axis=1))
print(f"Missing stalk-root values:\n{dataset.features.isna().sum().to_string()}")
print(f"Duplicate values: {int(dataset.features.duplicated().sum())}")
for feature in dataset.features:
    print(dataset.features[feature].value_counts().to_string(),"\n")
print(dataset.targets["poisonous"].value_counts().to_string())

#preprocessing
x = dataset.features
y = dataset.targets["poisonous"]
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=64,test_size=.25)
pipeline = make_pipeline(SimpleImputer(strategy="constant",fill_value="g"))
pipeline.steps.append(("encoder",OrdinalEncoder()))

#baseline
baseline_pipeline = clone(pipeline)
baseline_pipeline.steps.append(("classifier",RandomForestClassifier(random_state=42,max_features=22,n_jobs=-1)))
baseline_pipeline.fit(x_train,y_train)

#candidate pool culled models
feature_pool_culled = []
for i in range(22):
    new_pipeline = clone(pipeline)
    new_pipeline.steps.append(("classifier",RandomForestClassifier(random_state=42,max_features=22-i,n_jobs=-1)))
    feature_pool_culled.append(new_pipeline)

#feature culled models
specific_feature_removed = {}
for rm_feature in feature_list:
    culled_list = feature_list.copy()
    culled_list.remove(rm_feature)
    train_subset = x_train[culled_list]
    specific_feature_removed[rm_feature] = train_subset

#baseline evaluation
print("\nBaseline: ")
print(f"Accuracy: {baseline_pipeline.score(x_test,y_test)}")
test_predict = baseline_pipeline.predict(x_test)
print(f"Average Precision Score: {average_precision_score(y_test=='e',test_predict=='e')}")
print(f"F1 Score: {f1_score(y_test=='e',test_predict=='e')}")