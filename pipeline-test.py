import pyspark.sql.functions as f
import pyspark.ml as ml
from pyspark.sql.types import ArrayType, FloatType, IntegerType
import fungiclef.transforms as trans
from fungiclef.utils import get_spark


spark = get_spark(**{
    "spark.sql.parquet.enableVectorizedReader": False, 
})

# Get list of stored filed in cloud bucket
gcs_parquet_path = "gs://dsgt-clef-fungiclef-2024/dev/train/"
input_folder = f"dev_train.parquet"

dev_df = spark.read.parquet(gcs_parquet_path+input_folder)
#dev_df = dev_df.repartition('im_bytes')
dev_df.printSchema()
dev_df.count()

# Get subset of images to test pipeline
dev_subset_df = (
    dev_df.sample(fraction=0.0015, seed=42)
    .cache()
)
print(dev_subset_df.count())

# Init DINOv2 wrapper
dino = trans.WrappedDinoV2(input_col="im_bytes", output_col="transformed_data")

# Init Descrite Cosine Transform wrapper
dctn = trans.DCTN(input_col="transformed_data", output_col="dctn_data")

# Create Pipeline
pipeline = ml.Pipeline(stages=[dino]) #, dctn

# Fit pipeline to DF
model = pipeline.fit(dev_subset_df)

# Apply the model to transform the DF

transformed_df = model.transform(dev_subset_df).cache()

# Show results
transformed_df.select(["ImageUniqueID", "species", "transformed_data"]).show(n=10)
