# Databricks notebook source
# MAGIC %md
# MAGIC #IMPORTING LIBRARIES 

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql import functions as  F
import pandas as pd
import numpy as np 
import datetime 

# COMMAND ----------

# MAGIC %md
# MAGIC ### Generating the Customer and Vehicle IDs

# COMMAND ----------

num_customers = 1000000
customer_df = spark.range(1,num_customers + 1).withColumnRenamed("id","Customer_ID")
customer_df = customer_df.withColumn("Vehicle_ID", F.col("Customer_ID"))
display(customer_df)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Assigning policy tenure

# COMMAND ----------

tenure_dist = [0.2,0.3,0.4,0.1]
tenure_values = [1,2,3,4]

customer_df = customer_df.withColumn(
    "Policy_Tenure",
    F.when(F.col("Customer_ID") <= num_customers * tenure_dist[0], tenure_values[0])
     .when(F.col("Customer_ID") <= num_customers * (tenure_dist[0] + tenure_dist[1]), tenure_values[1])
     .when(F.col("Customer_ID") <= num_customers * (tenure_dist[0] + tenure_dist[1] + tenure_dist[2]), tenure_values[2])
     .otherwise(tenure_values[3])
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Assigning Purchase Dates and Calculate Other Fields 

# COMMAND ----------

import datetime
import pyspark.sql.functions as F
from pyspark.sql.types import StringType

start_date = datetime.date(2024, 1, 1)

num_customers = customer_df.count()
policies_per_day = num_customers // 365

def assign_purchase_date(customer_id):
    day_offset = (customer_id - 1) // policies_per_day
    return (start_date + datetime.timedelta(days=day_offset)).isoformat()

assign_purchase_date_udf = F.udf(assign_purchase_date, StringType())

customer_df = customer_df.withColumn(
    "Policy_Purchase_Date",
    assign_purchase_date_udf(F.col("Customer_ID"))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Calculating start/end date and premium

# COMMAND ----------

customer_df = customer_df.withColumn("Policy_Start_Date",
                                     F.date_add(F.col("Policy_Purchase_Date"), 365)
                                     )
customer_df = customer_df.withColumn("Policy_End_Date",
                                     F.expr("add_months(Policy_Start_Date, Policy_Tenure * 12)")
                                     )
customer_df = customer_df.withColumn("Premium",F.col("Policy_Tenure")*100
                                     )

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Filtering vehicles purchased on specific dates
# MAGIC

# COMMAND ----------

customer_df = customer_df.withColumn("Purchase_day", F.dayofmonth(F.col("Policy_Purchase_Date")))
claim_eligible_df = customer_df.filter(F.col("Purchase_Day").isin([7,14,21,28]))

# COMMAND ----------

claim_2025_df = claim_eligible_df.sample(withReplacement=False, fraction=0.3, seed=42)



# COMMAND ----------

# MAGIC %md
# MAGIC ### Creating Claim Data 

# COMMAND ----------

claim_2025_df = claim_2025_df.withColumn("Claim_Amount", F.lit(10000))
claim_2025_df = claim_2025_df.withColumn("Claim_Date", F.col("Policy_Start_Date"))
claim_2025_df = claim_2025_df.withColumn("Claim_Type", F.lit(1))

# COMMAND ----------

# MAGIC %md
# MAGIC ###Filter vehicles with 4 year policy active in 2026

# COMMAND ----------

four_year_policies_df = customer_df.filter(F.col("Policy_Tenure")==4)
claim_start_date = datetime.date(2026,1 ,1)
claim_end_date = datetime.date(2026,2,28)
days_in_period = (claim_end_date - claim_start_date).days + 1

# COMMAND ----------

claim_2026_candidates_df = four_year_policies_df.sample(withReplacement=False, fraction=0.1, seed=42)


# COMMAND ----------

# MAGIC %md
# MAGIC ### distributing claims evenly across the 59 days 

# COMMAND ----------

def assign_claim_date():
    day_offset = np.random.randint(0, days_in_period)
    return(claim_start_date + datetime.timedelta(days=int(day_offset))).isoformat()

assign_claim_date_udf = F.udf(assign_claim_date)

claim_2026_df = claim_2026_candidates_df.withColumn("Claim_Date", assign_claim_date_udf())

# COMMAND ----------

claim_2026_df = claim_2026_df.withColumn("Claim_Amount", F.lit(10000))
claim_2026_df = claim_2026_df.withColumn("Claim_Type", F.lit(2))

# COMMAND ----------

# MAGIC %md
# MAGIC #DATA ANALYSIS 

# COMMAND ----------

# MAGIC %md
# MAGIC ####Question 1 Calculate Total Premium Collected in 2024

# COMMAND ----------

# MAGIC %md
# MAGIC This analysis calculates the total premium revenue generated from all policies sold during the year 2024. Each policy has a premium amount determined by its tenure, where the premium is ₹100 multiplied by the number of years in the policy tenure. By summing the premium values across all policies sold in 2024, we estimate the total revenue collected by the insurance company from policy sales during that year. This metric is important because it represents the initial revenue base against which future claims and profitability will be evaluated.

# COMMAND ----------

total_premium_2024 = customer_df.agg(F.sum("Premium").alias("Total_Premium_2024"))
total_premium_2024.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ####Question 2  Calculate Total Claim Cost For Every Year (2025 and 2026) With Monthly Breakdown ?

# COMMAND ----------

# MAGIC %md
# MAGIC This analysis calculates the total claim costs incurred by the insurance company during 2025 and 2026, with the results grouped by month. Since claims occur based on predefined conditions (manufacturing defects and tenure rules), aggregating the claim amounts by month helps identify temporal patterns in claim activity. A monthly breakdown allows the company to understand when claims are most frequent and helps in cash flow planning, risk monitoring, and operational readiness.

# COMMAND ----------


claims_df = claim_2025_df.unionByName(claim_2026_df)

claims_df = claims_df.withColumn("Claim_Year", F.year("Claim_Date"))
claims_df = claims_df.withColumn("Claim_Month", F.month("Claim_Date"))


total_claim_cost = claims_df.groupBy("Claim_Year", "Claim_Month").agg(F.sum("Claim_Amount").alias("Total_Claim_Cost"))
total_claim_cost = total_claim_cost.orderBy("Claim_Year", "Claim_Month")
total_claim_cost.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ####Question 3 Calculate the claim cost to premium ratio for each policy tenure(1,2,3,4 years) ?

# COMMAND ----------

# MAGIC %md
# MAGIC This analysis calculates the claim cost to premium ratio for each policy tenure (1-year, 2-year, 3-year, and 4-year policies). The ratio is computed by dividing the total claim amount associated with policies of a particular tenure by the total premium collected from those policies. This metric helps evaluate the profitability of different policy durations. A lower ratio indicates that the premium collected is significantly higher than the claims paid, suggesting better profitability for that policy tenure.

# COMMAND ----------

claims_with_policies_df = claims_df.join(
    customer_df,
    ["Customer_ID","Vehicle_ID"],
    "inner"
).select(
    claims_df["Customer_ID"],
    claims_df["Vehicle_ID"],
    customer_df["Policy_Tenure"],
    claims_df["Claim_Amount"],
    customer_df["Premium"]
)

# COMMAND ----------

tenure_ratio_df = claims_with_policies_df.groupBy("Policy_Tenure").agg(
    (F.sum("Claim_Amount") / F.sum("Premium")).alias("Claim_Cost_to_Premium_Ratio")
)

tenure_ratio_df = tenure_ratio_df.orderBy("Policy_Tenure")

tenure_ratio_df.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ###Question 4 Calculate the claim cost to premium ratio by month of policy sale ?
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC This analysis examines how the month in which a policy was purchased affects the claim cost to premium ratio. Policies sold in different months may experience different claim patterns depending on claim eligibility rules and policy start dates. By grouping policies by their purchase month and computing the claim ratio for each month, the analysis helps identify seasonal patterns or risk concentration related to policy sales timing. This insight can help insurance companies optimize marketing strategies or adjust pricing based on risk patterns.

# COMMAND ----------

from pyspark.sql import functions as F

customer_df = customer_df.withColumn(
    "Purchase_Month",
    F.month("Policy_Purchase_Date")
)
policies_2024_df = customer_df.filter(
    F.year("Policy_Purchase_Date") == 2024
)

# COMMAND ----------

claims_with_policies_2024_df = claims_df.join(
    policies_2024_df,
    ["Customer_ID","Vehicle_ID"],
    "inner"
).select(
    claims_df["Customer_ID"],
    claims_df["Vehicle_ID"],
    policies_2024_df["Purchase_Month"],
    claims_df["Claim_Amount"],
    policies_2024_df["Premium"].alias("Policy_Premium")
)

display(claims_with_policies_2024_df)

# COMMAND ----------

month_ratio_2024_df = claims_with_policies_2024_df.groupBy("Purchase_Month").agg(
    (F.sum("Claim_Amount") / F.sum("Policy_Premium"))
    .alias("Claim_Cost_to_Premium_Ratio")
).orderBy("Purchase_Month")

display(month_ratio_2024_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Question 5 If every vehicle that has not yet made a claim eventually files exactly one claim during the remaining claiming policy tenure, estimate the total potential claim liability ?

# COMMAND ----------

# MAGIC %md
# MAGIC This analysis estimates the future claim liability for policies that have not yet filed a claim but may do so before their policy tenure expires. The assumption is that every vehicle that has not yet filed a claim will eventually file exactly one claim during the remaining active policy period. By identifying such policies and estimating their potential claim amounts, we calculate the total expected future claims the company may need to pay. This estimate helps the company understand its future financial obligations and risk exposure.

# COMMAND ----------

vehicles_with_claims = claims_df.select("Customer_ID","Vehicle_ID").distinct()

vehicles_without_claims = customer_df.join(vehicles_with_claims, ["Customer_ID","Vehicle_ID"], "left_anti")

num_vehicles_without_claims = vehicles_without_claims.count()

claim_amount_per_vehicle = 10000
total_potential_liability = num_vehicles_without_claims * claim_amount_per_vehicle

print(f"Estimated total potential claim liability: ₹{total_potential_liability}")

# COMMAND ----------

# MAGIC %md
# MAGIC ###Question 6  Assume daily premium = Total Premium ÷ Total Policy Tenure Days. Based on this: 
# MAGIC
# MAGIC

# COMMAND ----------

from pyspark.sql import functions as F

# Calculate total premium collected
total_premium = customer_df.agg(F.sum("Premium").alias("Total_Premium")).collect()[0]["Total_Premium"]

# Calculate total policy tenure days
# Sum up the product of each customer’s tenure by 365 days
customer_df = customer_df.withColumn("Policy_Tenure_Days", F.col("Policy_Tenure") * 365)
total_policy_tenure_days = customer_df.agg(F.sum("Policy_Tenure_Days").alias("Total_Tenure_Days")).collect()[0]["Total_Tenure_Days"]

# Calculate daily premium
daily_premium = total_premium / total_policy_tenure_days

# COMMAND ----------

# MAGIC %md
# MAGIC ####1 • Calculate the premium already earned by the company up to February 28, 2026 ?
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC This calculation determines how much premium revenue the company has already earned from the policies up to February 28, 2026. Since insurance revenue is earned gradually over the life of the policy, the daily premium is calculated by dividing the total premium by the total number of days in the policy tenure. The number of days elapsed since each policy started is then multiplied by the daily premium to estimate the premium already earned. This metric is important for understanding realized revenue versus remaining unearned premium.

# COMMAND ----------

from datetime import date
from pyspark.sql.types import IntegerType

# Calculate the days passed up to Feb 28, 2026
# For each policy, calculate days from purchase date to Feb 28, 2026
current_date = date(2026, 2, 28)

# Define a UDF to calculate days from purchase to Feb 28, 2026
def days_elapsed(purchase_date):
    purchase_dt = date.fromisoformat(purchase_date)
    return (current_date - purchase_dt).days

days_elapsed_udf = F.udf(days_elapsed, IntegerType())

# Apply the UDF to calculate days elapsed for each policy
customer_df = customer_df.withColumn("Days_Elapsed", days_elapsed_udf(F.col("Policy_Purchase_Date")))

# Calculate premium earned so far by summing daily premium times days elapsed
premium_earned_so_far = customer_df.agg(F.sum(F.col("Days_Elapsed") * daily_premium).alias("Premium_Earned_Upto_Feb28")).collect()[0]["Premium_Earned_Upto_Feb28"]

# COMMAND ----------

# MAGIC %md
# MAGIC ####2  • Estimate the premium expected to be earned monthly for the remaining policy period (assume 46 months remaining) ?
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC This analysis estimates the future premium revenue expected to be earned during the remaining policy period, assuming approximately 46 months remain. Using the daily premium calculation from the previous step, the remaining premium revenue can be estimated and distributed across the remaining months. This projection helps the insurance company forecast future revenue streams and plan financial resources accordingly.

# COMMAND ----------

# Estimate monthly premium for the remaining period
monthly_premium_estimate = daily_premium * 30  # Average days per month
expected_monthly_premium_46_months = monthly_premium_estimate  # Same amount each month for 46 months

print(f"Premium earned up to Feb 28, 2026: ₹{premium_earned_so_far}")
print(f"Expected monthly premium for the next 46 months: ₹{expected_monthly_premium_46_months}")

# COMMAND ----------

# MAGIC %md
# MAGIC ###BONUS QUESTIONS

# COMMAND ----------

# MAGIC %md
# MAGIC ###Question 1. Identify which policy tenure appears most profitable and explain why ?
# MAGIC

# COMMAND ----------

from pyspark.sql.functions import sum

tenure_profit_df = claims_df.join(
    customer_df,
    ["Customer_ID", "Vehicle_ID"],
    "inner"
).select(
    customer_df["Policy_Tenure"],
    claims_df["Claim_Amount"],
    customer_df["Premium"]
).groupBy("Policy_Tenure").agg(
    sum("Premium").alias("total_premium"),
    sum("Claim_Amount").alias("total_claim_cost")
).withColumn(
    "profit",
    (F.col("total_premium") - F.col("total_claim_cost"))
).orderBy("Policy_Tenure")

display(tenure_profit_df)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC
# MAGIC 1-Year policies appear most profitable because:
# MAGIC
# MAGIC 1️ They have short exposure to risk
# MAGIC 	•	The insurance company covers the vehicle only for 1 year.
# MAGIC
# MAGIC 2️ Lower probability of claims
# MAGIC 	•	The shorter the policy duration, the less time the customer has to file a claim.
# MAGIC
# MAGIC 3️ Premium is collected upfront
# MAGIC 	•	Even if a claim does not occur, the company keeps the premium.
# MAGIC
# MAGIC 4️ Longer tenure policies carry higher risk
# MAGIC 	•	3-year and 4-year policies remain active for a longer time, increasing the chance of claims.
# MAGIC
# MAGIC
# MAGIC After calculating the claim-to-premium ratio by policy tenure, I observed that shorter tenure policies generally have a lower loss ratio. This happens because the insurer is exposed to risk for a shorter period of time, reducing the probability of claims. Therefore, the 1-year policy appears to be the most profitable.

# COMMAND ----------

# MAGIC %md
# MAGIC ###Question 3 Estimate the loss ratio (Claims ÷ Premium) for the portfolio ?

# COMMAND ----------

from pyspark.sql.functions import sum

portfolio_ratio_df = claims_df.join(
    customer_df,
    ["Customer_ID", "Vehicle_ID"],
    "inner"
).select(
    claims_df["Claim_Amount"],
    customer_df["Premium"]
)

loss_ratio = portfolio_ratio_df.agg(
    (sum("Claim_Amount") / sum("Premium")).alias("portfolio_loss_ratio")
)

display(loss_ratio)

# COMMAND ----------

# MAGIC %md
# MAGIC The loss ratio measures the overall profitability of the insurance portfolio. It is calculated as the ratio of total claims paid to the total premium collected. A lower loss ratio indicates higher profitability, while a higher ratio suggests greater claim costs relative to premium revenue.

# COMMAND ----------

# MAGIC %md
# MAGIC ###Question 4  If claim frequency increases by 5% annually, estimate the impact on future profitability ?
# MAGIC

# COMMAND ----------

from pyspark.sql.functions import sum

claims_premium_df = claims_df.join(
    customer_df,
    ["Customer_ID", "Vehicle_ID"],
    "inner"
).select(
    claims_df["Claim_Amount"],
    customer_df["Premium"]
)

future_impact_df = claims_premium_df.agg(
    sum("Claim_Amount").alias("current_claim_cost"),
    sum("Premium").alias("total_premium")
).withColumn(
    "future_claim_cost",
    F.col("current_claim_cost") * 1.05
).withColumn(
    "future_loss_ratio",
    F.col("future_claim_cost") / F.col("total_premium")
)

display(future_impact_df)

# COMMAND ----------

# MAGIC %md
# MAGIC This analysis estimates how profitability may change if claim frequency increases by 5% annually. By adjusting the expected number of claims and recalculating the projected claim costs, we can estimate how the loss ratio and overall profitability may evolve in the future. This scenario analysis helps insurance companies assess risk exposure and plan pricing or policy adjustments to maintain profitability.

# COMMAND ----------

# MAGIC %md
# MAGIC ###ADDITIONAL TABLES FOR INSIGHTS 

# COMMAND ----------

from pyspark.sql.functions import year, month, sum

monthly_claims_df = claims_df \
    .withColumn("year", year("Claim_Date")) \
    .withColumn("month", month("Claim_Date")) \
    .groupBy("year", "month") \
    .agg(sum("Claim_Amount").alias("total_claim_cost")) \
    .orderBy("year", "month")

display(monthly_claims_df)

# COMMAND ----------

from pyspark.sql.functions import sum

tenure_ratio_df = claims_df.join(
    customer_df,
    ["Customer_ID", "Vehicle_ID"],
    "inner"
).select(
    customer_df["Policy_Tenure"],
    claims_df["Claim_Amount"],
    customer_df["Premium"]
).groupBy("Policy_Tenure") \
 .agg((sum("Claim_Amount") / sum("Premium"))
 .alias("claim_cost_to_premium_ratio")) \
 .orderBy("Policy_Tenure")

display(tenure_ratio_df)

# COMMAND ----------

from pyspark.sql.functions import sum

purchase_month_ratio_df = claims_df.join(
    customer_df,
    ["Customer_ID", "Vehicle_ID"],
    "inner"
).select(
    customer_df["Purchase_Month"],
    claims_df["Claim_Amount"],
    customer_df["Premium"]
).groupBy("Purchase_Month") \
 .agg((sum("Claim_Amount") / sum("Premium"))
 .alias("claim_cost_to_premium_ratio")) \
 .orderBy("Purchase_Month")

display(purchase_month_ratio_df)

# COMMAND ----------

display(customer_df)  #POLICY SALES DATA

# COMMAND ----------

display(claims_df)  # CLAIMS DATA

# COMMAND ----------

