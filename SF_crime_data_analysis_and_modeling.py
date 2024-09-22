# Databricks notebook source
# MAGIC %md
# MAGIC ## SF crime data analysis and modeling

# COMMAND ----------

# MAGIC %md 
# MAGIC ### In this notebook, you can learn how to use Spark SQL for big data analysis on SF crime data. (https://data.sfgov.org/Public-Safety/Police-Department-Incident-Reports-Historical-2003/tmnf-yvry). 
# MAGIC
# MAGIC

# COMMAND ----------

# DBTITLE 1,Import package 
from csv import reader
from pyspark.sql import Row 
from pyspark.sql import SparkSession
from pyspark.sql.types import *
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import warnings

import os
os.environ["PYSPARK_PYTHON"] = "python3"


# COMMAND ----------

# 从SF gov download data
#import urllib.request
#urllib.request.urlretrieve("https://data.sfgov.org/api/views/tmnf-yvry/rows.csv?accessType=DOWNLOAD", "/tmp/myxxxx.csv")
#dbutils.fs.mv("file:/tmp/myxxxx.csv", "dbfs:/laioffer/spark_hw1/data/sf_06_02.csv")
#display(dbutils.fs.ls("dbfs:/laioffer/spark_hw1/data/"))
## do it yourself
# https://data.sfgov.org/api/views/tmnf-yvry/rows.csv?accessType=DOWNLOAD


# COMMAND ----------

data_path = "dbfs:/laioffer/spark_hw1/data/sf_06_02.csv"
# use this file name later

# COMMAND ----------

# MAGIC %md
# MAGIC ### Solove  big data issues via Spark
# MAGIC approach: use SQL (recomend for data analysis or DS)  

# COMMAND ----------

# DBTITLE 1,Get dataframe and sql

from pyspark.sql import SparkSession
spark = SparkSession \
    .builder \
    .appName("crime analysis") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

df_opt1 = spark.read.format("csv").option("header", "true").load(data_path)
display(df_opt1)
df_opt1.createOrReplaceTempView("sf_crime")

## helper function to transform the date, choose your way to do it. 
# refer: https://jaceklaskowski.gitbooks.io/mastering-spark-sql/spark-sql-functions-datetime.html
# 方法1 使用系统自带udf
# from pyspark.sql.functions import to_date, to_timestamp, hour
# df_opt1 = df_opt1.withColumn('Date', to_date(df_opt1.OccurredOn, "MM/dd/yy"))
# df_opt1 = df_opt1.withColumn('Time', to_timestamp(df_opt1.OccurredOn, "MM/dd/yy HH:mm"))
# df_opt1 = df_opt1.withColumn('Hour', hour(df_opt1['Time']))
# df_opt1 = df_opt1.withColumn("DayOfWeek", date_format(df_opt1.Date, "EEEE"))

## 方法2 手工写udf 
#from pyspark.sql.functions import col, udf
#from pyspark.sql.functions import expr
#from pyspark.sql.functions import from_unixtime

#date_func =  udf (lambda x: datetime.strptime(x, '%m/%d/%Y'), DateType())
#month_func = udf (lambda x: datetime.strptime(x, '%m/%d/%Y').strftime('%Y/%m'), StringType())

#df = df_opt1.withColumn('month_year', month_func(col('Date')))\
#           .withColumn('Date_time', date_func(col('Date')))

## 方法3 手工在sql 里面
# select Date, substring(Date,7) as Year, substring(Date,1,2) as Month from sf_crime


## 方法4: 使用系统自带
# from pyspark.sql.functions import *
# df_update = df_opt1.withColumn("Date", to_date(col("Date"), "MM/dd/yyyy")) ##change datetype from string to date
# df_update.createOrReplaceTempView("sf_crime")
# crimeYearMonth = spark.sql("SELECT Year(Date) AS Year, Month(Date) AS Month, FROM sf_crime")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Q1 question (OLAP): 
# MAGIC #####Write a Spark program that counts the number of crimes for different category.
# MAGIC
# MAGIC Below are some example codes to demonstrate the way to use Spark RDD, DF, and SQL to work with big data. You can follow this example to finish other questions. 
# MAGIC

# COMMAND ----------

# DBTITLE 1,Spark dataframe based solution for Q1
q1_result = df_opt1.groupBy('category').count().orderBy('count', ascending=False)
display(q1_result)

# COMMAND ----------

# Visualize Result
import seaborn as sns
fig_dims = (20,6)
fig = plt.subplots(figsize=fig_dims)
q1_result_plot = q1_result.toPandas()
chart = sns.barplot(x = 'category', y = 'count', palette= 'BuGn_r',data = q1_result_plot )
chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right')

# COMMAND ----------

# DBTITLE 1,Spark SQL based solution for Q1
#Spark SQL based
spark_sql_q1 = spark.sql("SELECT  category, COUNT(*) AS Count FROM sf_crime GROUP BY category ORDER BY Count DESC")
display(spark_sql_q1)

# COMMAND ----------

# DBTITLE 1,Visualize your results
# important hints: 
## first step: spark df or sql to compute the statisitc result 
## second step: export your result to a pandas dataframe. 

crimes_pd_df = spark_sql_q1.toPandas()

# Spark does not support this function, please refer https://matplotlib.org/ for visuliation. You need to use display to show the figure in the databricks community. 

#display(p)

# COMMAND ----------

# MAGIC %md
# MAGIC Q1_Insight: According to the number of crimes, we can classify crime category into three groups based on the above-mentioned graphs and tables: high crime rate, medium crime rate, and low crime rate.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Q2 question (OLAP)
# MAGIC Counts the number of crimes for different district, and visualize your results
# MAGIC

# COMMAND ----------

spark_sql_q2 = spark.sql("SELECT PdDistrict, COUNT(*) AS Count FROM sf_crime GROUP BY 1 ORDER BY 2 DESC")
display(spark_sql_q2)

# COMMAND ----------

# df way
spark_df_q2 = df_opt1.groupBy('PdDistrict').count().orderBy('Count', ascending=False)
display(spark_df_q2)

# COMMAND ----------

crimes_dis_pd_df = spark_sql_q2.toPandas()
plt.figure()

ax = crimes_dis_pd_df.plot(kind = 'bar',x='PdDistrict',y = 'Count',logy= True,color = 'blue',legend = False, align = 'center')
ax.set_ylabel('count',fontsize = 12)
ax.set_xlabel('district',fontsize = 12)
plt.xticks(fontsize=8, rotation=30)
plt.title('#2 Number of crimes for different districts')
display()

# COMMAND ----------

# Visualize Result
fig_dims = (10,4)
fig = plt.subplots(figsize=fig_dims)
spark_df_q2_plot = spark_df_q2.toPandas()
chart = sns.barplot(x = 'PdDistrict', y = 'count', palette= 'BuGn_r',data = spark_df_q2_plot )
chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right')

# COMMAND ----------

# MAGIC %md
# MAGIC ###Q3 question
# MAGIC
# MAGIC Count the number of crimes each "Sunday" at "SF downtown".
# MAGIC hint 1: SF downtown is defiend via the range of spatial location. For example, you can use a rectangle to define the SF downtown, or you can define a cicle with center as well. Thus, you need to write your own UDF function to filter data which are located inside certain spatial range. You can follow the example here: https://changhsinlee.com/pyspark-udf/
# MAGIC
# MAGIC hint 2: SF downtown 物理范围可以是 rectangle a < x < b and c < y < d. thus, San Francisco Latitude and longitude coordinates are: 37.773972, -122.431297. X and Y represents each. So we assume SF downtown spacial range: X (-122.4213,-122.4313), Y(37.7540,37.7740). 也可以是中心一个圈，距离小于多少算做downtown

# COMMAND ----------

spark_sql_q3 = spark.sql("""
                      with Sunday_dt_crime as(
                      select substring(Date,1,5) as Date,
                             substring(Date,7) as Year
                      from sf_crime
                      where (DayOfWeek = 'Sunday'
                             and -122.423671 < X
                             and X < 122.412497
                             and 37.773510 < Y
                             and Y < 37.782137)
                             )
                             
                      select Year, Date, COUNT(*) as Count
                      from Sunday_dt_crime
                      group by Year, Date
                      order by Year, Date
                      """)
display(spark_sql_q3)

# COMMAND ----------

df_opt2 = df_opt1[['IncidntNum', 'Category', 'Descript', 'DayOfWeek', 'Date', 'Time', 'PdDistrict', 'Resolution', 'Address', 'X', 'Y', 'Location']]
display(df_opt2)
df_opt2.createOrReplaceTempView("sf_crime")

# COMMAND ----------

# MAGIC %md ###Q3 Answer
# MAGIC The number of crimes at each Sunday at SF Downtown is shown above.
# MAGIC The SF Downtown is defined in a retangluar area where (-122.423671 ≤ X ≤ 122.412497) and (37.773510 ≤ Y ≤ 37.782137).

# COMMAND ----------

# MAGIC %md
# MAGIC #### Q4 question (OLAP)
# MAGIC Analysis the number of crime in each month of 2015, 2016, 2017, 2018. Then, give your insights for the output results. What is the business impact for your result?  

# COMMAND ----------

df_opt2.createOrReplaceTempView('sf_crime')

# COMMAND ----------

spark_sql_q4 = spark.sql("""
                       SELECT SUBSTRING(Date,1,2) AS Month, SUBSTRING(Date,7,4) AS Year, COUNT(*) AS Count
                       FROM sf_crime
                       GROUP BY Year, Month
                       HAVING Year in (2015, 2016, 2017, 2018) 
                       ORDER BY Year, Month
                       """)
display(spark_sql_q4)

# COMMAND ----------

# MAGIC %md
# MAGIC ####Q4 Answer
# MAGIC ####Analysis of the Updated Crime Data from 2015 to 2018
# MAGIC The provided bar chart visualizes the number of crimes per month for the years 2015, 2016, 2017, and 2018. Here are the detailed insights and business impacts based on the updated data:
# MAGIC
# MAGIC #### Analysis and Insights:
# MAGIC The crime numbers are relatively stable in year 2015, 2016, 2017. There is a significant reduction in crime numbers across all months in 2018, particularly noticeable in May with a count of only 3,519.
# MAGIC 1. **Impact of California Proposition 47:**
# MAGIC    - **Proposition 47:** Passed in 2014, this act reclassified certain non-violent offenses, including theft and drug possession, from felonies to misdemeanors. This legislative change likely led to an increase in reported thefts and other petty crimes, contributing to higher crime rates observed from 2015 to 2017.
# MAGIC 2. **Reasons for Decline in 2018:**
# MAGIC    - **Increased Police Patrols:** Enhanced uniformed police presence in 2018 likely deterred criminal activities, contributing to the observed reduction in crime rates, particularly significant in May.
# MAGIC    - **Crackdown on Drug Trade:** The San Francisco Police Department's intensified efforts to combat the drug trade likely played a crucial role in reducing crime rates. Drug-related activities often contribute to theft, violence, and other crimes, and addressing the drug trade would have a broad impact on overall crime statistics.
# MAGIC
# MAGIC #### Business Impact:
# MAGIC 1. **Policy Making:**
# MAGIC    - Policymakers can use this data to implement targeted crime prevention strategies during high-crime months. For instance, community outreach programs and crime awareness campaigns could be intensified in the first quarter of the year.
# MAGIC
# MAGIC 2. **Budget Planning:**
# MAGIC    - The data provides a basis for budgeting for public safety. Knowing that certain months require more resources, budget allocations can be adjusted accordingly to ensure adequate funding for high-demand periods.
# MAGIC
# MAGIC 3. **Long-Term Strategies:**
# MAGIC    - The overall reduction in crime from 2015 to 2018 indicates that existing strategies might be working. Continued investment in successful programs, coupled with data-driven adjustments, can sustain or further reduce crime rates.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Q5 question (OLAP)
# MAGIC Analysis the number of crime with respsect to the hour in certian day like 2015/12/15, 2016/12/15, 2017/12/15. Then, give your travel suggestion to visit SF. 

# COMMAND ----------

# Show number of crime by hour for all records
spark_sql_q5 = spark.sql("""
                      select substring(Time,1,2) as Hour,
                      count(*) as Count
                      from sf_crime
                      group by Hour
                      order by Hour
                      """)
display(spark_sql_q5)

# COMMAND ----------

# Show number of crime by hour for records in Christmas
spark_sql_q5 = spark.sql("""
                      select substring(Time,1,2) as Hour,
                      count(*) as Count
                      from sf_crime
                      where Date like '12/25/%'
                      group by Hour
                      order by Hour
                      """)
display(spark_sql_q5)

# COMMAND ----------

# MAGIC %md
# MAGIC advice: travel to SF on on Christmas Day from 9-11 am and 2-6 PM, because there has been a marked decrease in the number of crimes.

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Q6 question (OLAP)
# MAGIC (1) Step1: Find out the top-3 danger disrict  
# MAGIC (2) Step2: find out the crime event w.r.t category and time (hour) from the result of step 1  
# MAGIC (3) give your advice to distribute the police based on your analysis results. 
# MAGIC

# COMMAND ----------

spark_sql_q6_s1 = spark.sql( """
                             SELECT PdDistrict, COUNT(*) as Count
                             FROM sf_crime
                             where PdDistrict is not NULL
                             GROUP BY 1
                             ORDER BY 2 DESC
                             LIMIT 3 
                             """ )
display(spark_sql_q6_s1)

# COMMAND ----------

# MAGIC %md
# MAGIC Q6 advice
# MAGIC According to step1, the three most dangerous districts are SOUTHERN, MISSION and NORTHERN.
# MAGIC We can see from the picture above that among the top three dangerous streets, the crime rate around 5 am is the lowest, and the high incidence of crime rate is around 12pm and 18pm, especially pay attention to theft, so I recommend to increase police patrol during that periods.

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Q7 question (OLAP)
# MAGIC For different category of crime, find the percentage of resolution. Based on the output, give your hints to adjust the policy.

# COMMAND ----------

res_num = spark.sql("""select category, resolution, count(*) as N_res from sf_crime group by category, resolution""")
res_num.createOrReplaceTempView("res_num")

cate_num = spark.sql("""select category, count(*) as N_cate from sf_crime group by category""")
cate_num.createOrReplaceTempView("cate_num")

q7_result = spark.sql("""
                      select distinct sf_crime.category, sf_crime.resolution, N_res/N_cate as Percentage
                      from (sf_crime left join res_num on sf_crime.category = res_num.category and sf_crime.resolution = res_num.resolution)
                      left join cate_num on sf_crime.category = cate_num.category
                      order by category, resolution""")
q7_result.createOrReplaceTempView("q7_result")

# COMMAND ----------

display(q7_result)

# COMMAND ----------

q7_result_2 = spark.sql("""
                      select distinct res_num.category, res_num.resolution, N_res/N_cate as Percentage
                      from res_num left join cate_num on res_num.category = cate_num.category
                      order by category, resolution""")
display(q7_result_2)

# COMMAND ----------

# Percentage of resolution for LARCENY/THEFT
q7 = spark.sql("""
               select Resolution, Percentage
               from q7_result
               where category = 'LARCENY/THEFT'
               order by Percentage desc
               """)
display(q7)

# COMMAND ----------

# Percentage of resolution for BURGLARY
q7 = spark.sql("""
               select Resolution, Percentage
               from q7_result
               where category = 'BURGLARY'
               order by Percentage desc
               """)
display(q7)

# COMMAND ----------

# Percentage of resolution for ASSAULT
q7 = spark.sql("""
               select Resolution, Percentage
               from q7_result
               where category = 'ASSAULT'
               order by Percentage desc
               """)
display(q7)

# COMMAND ----------

# MAGIC %md
# MAGIC Q7 advice
# MAGIC The percentage of resolution for LARCENY/THEFT, BURGLATY and ASSAULT are shown above.
# MAGIC Surprisingly, it shows that most crime cases are not resolved (with 'NONE' in resolution).
# MAGIC In addition to none resolution, the most common processing resolutions in top-3 danger disrict are arrest, booked and arrest, cited. For cases where there are too many none processing methods, the following suggestions are given. By focusing on improving law enforcement capabilities, enhancing community engagement, leveraging technology, and ensuring accountability, it is possible to increase the number of resolved cases, thereby improving public safety and trust in the justice system.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Conclusion. 
# MAGIC Story and Purpose: I conducted a comprehensive big data analysis on San Francisco crime data to uncover patterns and provide actionable insights for improving public safety.
# MAGIC
# MAGIC Technical Approach: Utilizing Spark and Spark SQL, I processed and analyzed a large dataset, applying techniques such as data cleaning and OLAP operations. The analysis included counting crimes by category and district, identifying crime trends over time, and analyzing crimes at specific locations and times.
# MAGIC
# MAGIC Learnings from Data: Through this analysis, I identified key crime trends, such as high crime rates in certain districts and time periods, and discovered that a significant portion of crimes, particularly larceny/theft, remain unresolved.     This led to advising on policy changes and targeted crime prevention strategies.
# MAGIC
# MAGIC Business Impact: The results provided critical insights for law enforcement and city planners, enabling more effective resource distribution and public safety measures. By identifying peak crime times and high-risk areas, the analysis also offered recommendations for safer travel, contributing to enhanced community safety and operational efficiency in addressing crime in San Francisco.
# MAGIC
# MAGIC
