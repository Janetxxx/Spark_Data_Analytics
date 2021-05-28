from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType,IntegerType, FloatType
import argparse

# create a SparkSession, The entry point to programming Spark with the Dataset and DataFrame API.
spark = SparkSession \
    .builder \
    .appName("workload1_Ass2") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

parser = argparse.ArgumentParser()
parser.add_argument("--output", help="the output path",
                        default='ass2_workload1_out')
args = parser.parse_args()
output_path = args.output

# load JSON file into Spark data frame 
tweets_df = spark.read.option("multiLine","true").json('tweets.json')
#tweets_df.show(1)

# WORKLOAD 1

# concat retweet_id and replyto_id within each retweet object, add into a new column 
import pyspark.sql.functions as F

def myConcat(*cols):
    return F.concat(*[F.coalesce(c, F.lit("")) for c in cols])

# create a new data frame contains only user_id with rt_rp_id.
df = tweets_df.withColumn("rt_rp_id", myConcat("retweet_id","replyto_id")).select("user_id","rt_rp_id")

# group by user_id and get document representation 
df_new = df.groupby("user_id").agg(F.collect_set("rt_rp_id")).withColumnRenamed("collect_set(rt_rp_id)", "Document_Representation")

from pyspark.sql.functions import *

dataframe = df_new.withColumn('Document_Representation', concat_ws(',', 'Document_Representation'))
dataframe.show() #complete version

# Apply the feature extractor TF-IDF on the document representations.
from pyspark.ml.feature import HashingTF, IDF, Tokenizer

tokenizer = Tokenizer(inputCol="Document_Representation", outputCol="words")
wordsData = tokenizer.transform(dataframe)
#wordsData.show(truncate=0)

# HashingTF is a Transformer, which converts wordsData into fixed-length feature vectors
hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=20)
featurizedData = hashingTF.transform(wordsData)
#featurizedData.show(truncate=0)

# IDF is an Estimator which is fit on a dataset and produces an IDFModel.
idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurizedData)
# The IDFModel takes feature vectors and scales each feature 
# down-weights features which appear frequently in a corpus
rescaledData = idfModel.transform(featurizedData)
feature_df = rescaledData.select("user_id","features")
feature_df.show(truncate=0)


from pyspark.sql.types import *
# convert sparse vector to normal vector
vector_udf = udf(lambda vector: vector.toArray().tolist(), ArrayType(DoubleType()))
vector_df = feature_df.select("user_id",vector_udf('features').alias('features'))

test_item = vector_df.take(1)
test_input = spark.createDataFrame(test_item)
test_input.cache()
test_input.show()

test_value = test_input.select("features").collect()[0][0]

# Calculate cosine similarity between Vectors and test_input
import numpy as np
def cos_similarity(vec):
    sim = np.dot(test_value,vec)/(np.linalg.norm(test_value)*np.linalg.norm(vec))
    return sim.tolist()
    
cos_sim_udf = udf(cos_similarity, FloatType())

df_cos = vector_df.withColumn('cos_sim', cos_sim_udf('features')).select("user_id","features","cos_sim")

# top 5 users with similar interest as a given user id, sorting by Descending order
top5users = df_cos.select('user_id','cos_sim').orderBy('cos_sim', ascending=False).limit(5).collect()
top_5_user_id = []
for x in top5users:
    top_5_user_id.append(x[0])
print("Using TF-IDF as feature extractor:")
print("The top 5 users with similar interest as a given user id:",top_5_user_id)



#######################################################################################
# Apply the feature extractor Word2Vec on the document representations.
from pyspark.ml.feature import Word2Vec

word2Vec = Word2Vec(vectorSize=10, minCount=0, inputCol="Document_Representation", outputCol="result")
model = word2Vec.fit(df_new)
# The Word2VecModel transforms each document into a vector using the average of all words in the document
result = model.transform(df_new)
Word2Vec_result = result.collect()

#result.show(truncate=0)
for x1 in Word2Vec_result:
    print("Document Representation: {} => \nVector: {}\n".format(x1[1], x1[2]))

# test data 
test_item1 = result.take(1)
test_input1 = spark.createDataFrame(test_item1)
test_input1.cache()
test_input1.show()
test_value1 = test_input1.select("result").collect()[0][0]
print(test_value1)

# Calculate the cosine similarities using the values obtained in previous step 
from pyspark.sql.functions import udf, log
import numpy as np
def cos_similarity(vec):
    sim = np.dot(test_value1,vec)/(np.linalg.norm(test_value1)*np.linalg.norm(vec))
    return sim.tolist()
    
cos_sim_udf1 = udf(cos_similarity, FloatType())

df_cos1 = result.withColumn('cos_sim_Word2Vec', cos_sim_udf1('result')).select("user_id","result","cos_sim_Word2Vec")
df_cos1.show()

# top 5 users with similar interest as a given user id, sorting by Descending order
top5users_Word2Vec = df_cos1.select('user_id','cos_sim_Word2Vec').orderBy('cos_sim_Word2Vec', ascending=False).limit(5).collect()
top_5_user_id_Word2Vec = []
for x in top5users_Word2Vec:
    top_5_user_id_Word2Vec.append(x[0])
print("Using Word2Vec as feature extractor:")
print("The top 5 users with similar interest as a given user id:", top_5_user_id_Word2Vec)



#########################################################################################################################
# WORKLOAD 2

# prepare the raw data in the format as required by the collaborative filter algorithm
# show user_id, user_mentions
df_wk2 = tweets_df.select("user_id","user_mentions")
#df_wk2.show(5)

from pyspark.sql.functions import col, explode

df_mention = df_wk2.withColumn("user_mentions", explode("user_mentions")) \
                .select("*", col("user_mentions")["id"].alias("mention_id")) \
                .select("user_id","mention_id")
df_mention.show(5)

# count rating: number of times a tweet user mentions a mention user.
from pyspark.sql.functions import *
df_rating = df_mention.groupBy("user_id","mention_id") \
            .agg(count("mention_id")).select("user_id","mention_id","count(mention_id)")

df_rating.show(5)

# prepare the raw data in the format as required by the collaborative filter algorithm
raw_data = df_rating.select("user_id", "mention_id", col("count(mention_id)").alias("rating"))
raw_data.show(5)
raw_data.cache()

raw_dataframe = raw_data.withColumn("user_id_INT", raw_data["user_id"].cast(IntegerType())) \
                        .withColumn("mention_id_INT", raw_data["mention_id"].cast(IntegerType()))

raw_dataframe.cache()

# Build a model to perform the recommendation.
# Alternating Least Squares (ALS) is a the model I'll use to fit my data and find similarities.
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row

(training, test) = raw_dataframe.randomSplit([0.8, 0.2])

# Build the recommendation model using ALS on the training data
# Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics
als = ALS(maxIter=5, regParam=0.01, userCol="user_id_INT", itemCol="mention_id_INT", ratingCol="rating",
          coldStartStrategy="drop")
model = als.fit(training)

predictions = model.transform(test)
predictions.show(5)

evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print("Root-mean-square error = " + str(rmse))

# Generate top 5 mentioned users recommendations for each tweeter user
Rec = model.recommendForAllUsers(5)

Recommendation = Rec.join(predictions, "user_id_INT").select("user_id","recommendations")
Recommendation.show(truncate=0)
# in recommendations sections, {mention_user_id, predicted_mentioned_times}

spark.stop()


